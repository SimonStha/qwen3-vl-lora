#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Loading libraries
"""
import transformers
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from PIL import Image
from trl import SFTTrainer, SFTConfig
import io
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
import os
import argparse
import wandb
from collections import defaultdict, Counter

print("All imports successful!")
print(transformers.__version__)
print(transformers.__file__)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL with LoRA on MIMIC-CXR")

parser.add_argument("--r",                  type=int,   default=32,     help="LoRA rank")
parser.add_argument("--lr",                 type=float, default=1e-4,   help="Learning rate")
parser.add_argument("--epochs",             type=int,   default=3,      help="Number of training epochs")
parser.add_argument("--batch_size",         type=int,   default=2,      help="Per-device train batch size")
parser.add_argument("--gradient_steps",     type=int,   default=2,     help="Gradient accumulation steps")
parser.add_argument("--warmup_ratio",       type=float, default=0.1,    help="Warmup ratio")
parser.add_argument("--lora_dropout",       type=float, default=0.05,    help="LoRA dropout")
parser.add_argument("--weight_decay",       type=float, default=0.01,   help="Weight decay")
parser.add_argument("--max_samples",        type=int,   default=13,     help="Max samples per phrase (dedup cap)")
parser.add_argument("--eval_steps",         type=int,   default=200,    help="Eval and save frequency")
parser.add_argument("--bf16",             action="store_true", help="Use bf16 precision")
parser.add_argument("--fp16",             action="store_true", help="Use fp16 precision")
parser.add_argument("--early_stopping",     type=int,   default=3,      help="Early stopping patience")
parser.add_argument("--model_name",         type=str,   default="Qwen/Qwen3-VL-4B-Instruct", help="Base model")
parser.add_argument("--dataset",            type=str,   default="itsanmolgupta/mimic-cxr-dataset", help="HF dataset")
parser.add_argument("--output_dir",         type=str,   default=None,   help="Override checkpoint output dir")

args = parser.parse_args()

# Precision logic: default to bf16 if neither specified, and ensure not both
if args.bf16 and args.fp16:
    parser.error("Cannot use both --bf16 and --fp16. Please choose one.")
if not args.bf16 and not args.fp16:
    args.bf16 = True

# Derived variables
r           = args.r
l_alpha     = r*2
lr          = args.lr
epochs      = args.epochs
batch_size  = args.batch_size
gradient_steps  = args.gradient_steps
warmup_ratio    = args.warmup_ratio
lora_dropout    = args.lora_dropout
weight_decay    = args.weight_decay
model_name      = args.model_name
MAX_SAMPLES_PER_PHRASE = args.max_samples
bf16 = args.bf16
fp16 = args.fp16
compute_dtype = torch.bfloat16 if bf16 else torch.float16
run_name   = f"Qwen3-VL-4B-Instruct-xray-lora-r{r}-lr{lr}-epochs{epochs}-batch{batch_size}"
output_dir = args.output_dir or f"./checkpoints/{run_name}"

# In[3]:


"""
load mimic-cxr dataset
"""
ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")

print(ds)

def has_impression(example):
    return example["impression"] is not None and len(example["impression"].strip()) > 0

filtered_ds = ds["train"].filter(has_impression)

# This is muy important
MAX_SAMPLES_PER_PHRASE = 13 # There are certain phrases in the dataset that appear most often and it's making it so that the model is not training on the right information
impression_tracker = defaultdict(int)
indices_to_keep = []

print(f"Capping exact impression matches to a maximum of {MAX_SAMPLES_PER_PHRASE}...")

# Iterate through the text column only (faster than iterating full examples)
for i, impression_text in enumerate(filtered_ds["impression"]):
    # Optional: strip whitespace and lowercase for stricter matching
    clean_text = impression_text.strip().lower() 

    if impression_tracker[clean_text] < MAX_SAMPLES_PER_PHRASE:
        indices_to_keep.append(i)
        impression_tracker[clean_text] += 1

# Filter the dataset to only keep the allowed indices
filtered_ds = filtered_ds.select(indices_to_keep)
# ==========================================

# filtered_ds = filtered_ds.shuffle(seed=42).select(range(20000)) # "uncomment to train on smaller dataset"

print("Original size:", len(ds["train"]))
print("Filtered & Capped size:", len(filtered_ds))

# Split the filtered dataset into train and temporary (test + validation)
# 80% Train, 20% Temp
train_test_valid_ds = filtered_ds.train_test_split(test_size=0.2, seed=42)

# Split the temporary set into validation and test (50% each of the 20% -> 10% of total each)
test_valid_ds = train_test_valid_ds['test'].train_test_split(test_size=0.5, seed=42)

# Combine into a single DatasetDict
# split_ds is not a dataset with train(80%), validation(10%), and test(10%)
split_ds = DatasetDict({
    'train': train_test_valid_ds['train'],
    'validation': test_valid_ds['train'],
    'test': test_valid_ds['test']
})

print("\nDataset structure after splitting:")
print(split_ds)

"""
After splitting the dataset, check how many of the most common phrases there are
"""
# List of common phrases radiologists use for healthy/stable X-rays
normal_phrases = [
    "no acute cardiopulmonary process",
    "no acute cardiopulmonary finding",
    "no acute cardiopulmonary abnormality",
    "clear",
    "unremarkable",
    "no substantial change",
    "as above",
    "no significant interval change",
    "normal"
]

def is_healthy(example):
    # Convert impression to lowercase for easy matching
    text = str(example['impression']).lower()
    # If any of the normal phrases are in the text, flag it as healthy
    return any(phrase in text for phrase in normal_phrases)

# Filter the dataset into two separate buckets
healthy_ds = split_ds['train'].filter(is_healthy)
abnormal_ds = split_ds['train'].filter(lambda x: not is_healthy(x))

print(f"\nTotal Training Samples: {len(split_ds['train'])}")
print(f"Healthy (Keyword Match): {len(healthy_ds)}")
print(f"Abnormal (Keyword Match): {len(abnormal_ds)}")

"""
Check what the 15 most common impressions are
"""
# Extract all impressions and count their exact occurrences
all_impressions = split_ds['train']['impression']
impression_counts = Counter(all_impressions)

print("\n--- Top 15 Most Common Impressions (Post-Capping) ---")
for text, count in impression_counts.most_common(15):
    # Calculate the percentage of the dataset this single phrase takes up
    percentage = (count / len(all_impressions)) * 100
    print(f"[{percentage:.1f}%] {count} samples: {text}")


# In[4]:


import torch

def collate_fn(examples):
    texts = []
    images = []

    for example in examples:
        img = example["images"][0]

        # Handle case where dataset serialized PIL Image to dict/bytes
        if isinstance(img, dict):
            img = Image.fromarray(img["array"]) if "array" in img else Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))

        # Ensure RGB
        img = img.convert("RGB")
        images.append(img)

        # Apply Qwen chat template
        text = processor.tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)

    # Process batch
    batch = processor(
        text=texts, 
        images=[[img] for img in images], 
        return_tensors="pt", 
        padding=True
    )

    labels = batch["input_ids"].clone()

    # 1. Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 2. Mask the User Prompt and Image Tokens
    # Get the token IDs for Qwen's assistant header. 
    # (add_special_tokens=False ensures we just get the raw tokens for matching)
    assistant_header = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    header_len = len(assistant_header)

    for i in range(labels.shape[0]):
        seq = labels[i].tolist()
        start_idx = -1

        # Search for the assistant header sequence in the tokenized input
        for j in range(len(seq) - header_len):
            if seq[j:j+header_len] == assistant_header:
                start_idx = j + header_len
                break

        if start_idx != -1:
            # Set everything before the actual answer (image tokens, system prompt, user prompt, etc.) to -100
            labels[i, :start_idx] = -100
        else:
            print(f"Warning: Could not find assistant header in sequence {i}. Loss calculation might be skewed.")

    batch["labels"] = labels
   
    return batch


# In[5]:


def qwen3vl_format(ex):
    return {
        "images": [ex["image"]],   # top-level list of PIL images
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},   # placeholder only, no "image" key
                    {"type": "text",
                     "text": (
                         "You are assisting a radiologist. "
                         "Based on the given chest X-ray, generate a concise clinical impression."
                     )}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Impression: {ex['impression']}"}],
            },
        ]
    }

# Apply the formatting function to the dataset
# .map makes it so that it efficiently applies the format to dataset
# remove_columns so that the original columns (image, findings, and impression) are discarded and only the formatted data is left
formatted_ds = split_ds.map(qwen3vl_format, remove_columns=['image', 'findings', 'impression'])

# Verify the format
print("First example from train set:")
print(formatted_ds['train'][0]['messages'])

os.environ["WANDB_PROJECT"] = "spring_2026_research"
os.environ["WANDB_NOTEBOOK_NAME"] = "/home/andy/finetuning-workspace/workspace/qwen3-vl-lora.ipynb"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_SAVE_CODE"] = "true"
os.environ["WANDB_WATCH"] = "false"
#wandb.login()

wandb.init(
    project=os.environ["WANDB_PROJECT"], 
    name=run_name,
    tags=["qwen3-vl", "lora", "mimic-cxr", "strix-halo"],
    group="r32_regularization_tests", 
    job_type="lora-adapter",
    config={
        "lora_rank": r,
        "lora_alpha": l_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_steps": gradient_steps,
        "warmup_ratio": warmup_ratio,
        "dataset_size": len(filtered_ds),
        "notes": "Testing lower rank and lower learning rate to try to prevent overfitting in previous adapter"
    }
)


# In[8]:


model_name = "Qwen/Qwen3-VL-4B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=compute_dtype,
    attn_implementation="eager",
)

processor = AutoProcessor.from_pretrained(model_name, use_fast=False) # Changed from AutoTokenizer to AutoProcessor
torch_dtype = model.dtype

# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different VLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=r,
    lora_alpha=l_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print(f"Weights footprint: {model.get_memory_footprint()/1e9:.2f} GB")


# In[10]:


# Configure training arguments using SFTConfig
training_args = SFTConfig(
    # Training schedule / optimization
    num_train_epochs=epochs,
    weight_decay=weight_decay,
    # max_steps=10,                                         # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    per_device_train_batch_size=batch_size,                        # Batch size per GPU/CPU
    gradient_accumulation_steps=gradient_steps,                        # Gradients are accumulated over multiple steps → effective batch size = 4 * 8 = 32
    warmup_ratio=warmup_ratio, #Gradually increase LR until the warmup ratio. Better than warmup_steps since it gets the ratio of your dataset. Useful if your dataset changes a lot
    learning_rate=lr,                                   # Learning rate for the optimizer
    optim="adamw_torch_fused",                                   # Optimizer
    max_length=None,                                      # For VLMs, truncating may remove image tokens, leading to errors during training. max_length=None avoids it
    packing=False,
    fp16=fp16,
    bf16=bf16,
    lr_scheduler_type="cosine",
    remove_unused_columns=False, #added
    run_name=run_name,
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    # Logging / reporting
    output_dir=output_dir,                                # Where to save model checkpoints and logs
    logging_steps=10,                                      # Log training metrics every N steps
    report_to="wandb",                                  # Experiment tracking tool

    eval_strategy="steps", # Evaluate model at end of each epoch
    eval_steps=200,
    save_strategy="steps", # Save a checkpoint at the end of each epoch
    save_steps=200,

    load_best_model_at_end=True, # Keep the best checkpoint so it doesn't just keep the last one
    metric_for_best_model="loss",
    greater_is_better=False
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_ds['train'],
    eval_dataset=formatted_ds['validation'],
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stops if val loss gets worse for 1 epoch
)


# In[ ]:


torch.cuda.reset_peak_memory_stats()
trainer.train()
trainer.save_model()
wandb.finish()


# # YOUR MODEL IS DONE TRAINING!!!!!!

# In[ ]:


def generate_impression(model, processor, image):
    prompt = (
        "You are assisting a radiologist. "
        "Based on the given chest X-ray, generate a concise clinical impression."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Apply Qwen chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Preprocess
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # deterministic output (better for evaluation)
        )

    # Remove input tokens
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    # Decode only new tokens
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return output.strip()


# In[14]:


from transformers import AutoModelForImageTextToText
from peft import PeftModel

"""
Loading VLM base model and processor
"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)


base = AutoModelForImageTextToText.from_pretrained(
    model_name, 
    quantization_config=bnb_config,
    device_map="auto", 
    attn_implementation="eager" 
)

processor = AutoProcessor.from_pretrained(model_name)
model = PeftModel.from_pretrained(base, output_dir)


# In[ ]:


formatted_ds['test'][0]


# In[ ]:


import random

for i in range(5):
    sample = formatted_ds['test'][random.randint(0,200)]
    image = sample["images"]
    gt = sample["messages"]

    pred = generate_impression(model, processor, image)

    print(f"\n=== Sample {i} ===")
    print("Ground Truth:\n", gt)
    print("\nFine-Tuned Prediction:\n", pred)


# In[17]:
from evaluate import load
from tqdm import tqdm

rouge = load("rouge")
bleu = load("bleu")

predictions = []
references = []

test_subset = formatted_ds['test'].select(range(min(100, len(formatted_ds['test']))))

for sample in tqdm(test_subset):
    image = sample["images"]

    # Ground truth extraction
    if "impression" in sample:
        gt = sample["impression"]
    else:
        gt = sample["messages"][-1]["content"]

    # Generate prediction
    pred = generate_impression(model, processor, image)

    # Clean text
    pred_clean = str(pred).replace("Impression:", "").strip()
    gt_clean = str(gt).strip()

    predictions.append(pred_clean)
    references.append(gt_clean)

# ROUGE
rouge_results = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True
)

# BLEU 
bleu_results = bleu.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

print("\n=== Final Test Set Scores ===")
print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"BLEU: {bleu_results['bleu']:.4f}")
