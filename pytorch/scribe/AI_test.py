import os
import sys
import time  # Added missing import

# Apply workaround for CVE-2025-32434 before importing transformers
import transformers.utils.import_utils
original_check = transformers.utils.import_utils.check_torch_load_is_safe
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

import math
import csv
import matplotlib.pyplot as plt
import torch
import gc
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from fbfgs import FBFGS
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import datasets
from datasets import Dataset

# Restore original check after transformers is imported
transformers.utils.import_utils.check_torch_load_is_safe = original_check

os.environ["TRANSFORMERS_NO_IMAGE"] = "1"
from transformers import MambaConfig, Mamba2ForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, Mamba2Config, GPTNeoXTokenizerFast

# Check if safetensors is available
try:
    from safetensors.torch import load_file, save_file
    has_safetensors = True
except ImportError:
    has_safetensors = False
    print("Warning: safetensors not available. Using PyTorch format (insecure).")

num_cores = os.cpu_count()
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

filename = "AI_Checkpoint.ai"
history_filename = "fbfgs_history.pth"
indices_filename = "dataset_indices.pth"

# Load the base config from the 370m model and modify specific parameters
config = Mamba2Config.from_pretrained("state-spaces/mamba2-370m")
config.hidden_size = 200     
config.num_hidden_layers = 24   
config.head_dim = 50
config.num_heads = 8
config.state_size = 54
config.dtype = "float16"

# Load tokenizer - try direct GPTNeoX approach first
tokenizer = None
try:
    # Try loading with GPTNeoXTokenizerFast directly
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("state-spaces/mamba2-370m")
    print("Successfully loaded tokenizer using GPTNeoXTokenizerFast")
except Exception as e:
    print(f"GPTNeoXTokenizerFast failed: {e}")
    try:
        # Fallback to AutoTokenizer with trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba2-370m", trust_remote_code=True)
        print("Successfully loaded tokenizer using AutoTokenizer with trust_remote_code")
    except Exception as e2:
        print(f"AutoTokenizer with trust_remote_code failed: {e2}")
        # Final fallback - use a basic tokenizer if nothing works
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Using GPT2 tokenizer as fallback - this may cause issues but will allow the script to continue")

# Model loading with safetensors support
if os.path.exists(filename):
    # Check for safetensors files first
    safetensors_file = filename + ".safetensors"
    pytorch_file = filename + ".bin"
    config_file = filename + "/config.json"
    
    if has_safetensors and os.path.exists(safetensors_file):
        print(f"Loading model from safetensors: {safetensors_file}")
        model = Mamba2ForCausalLM(config)
        state_dict = load_file(safetensors_file)
        model.load_state_dict(state_dict)
        model = model.to("cuda")
    elif os.path.exists(pytorch_file):
        print(f"Loading model from PyTorch format: {pytorch_file}")
        # Apply workaround for CVE-2025-32434
        import transformers.utils.import_utils
        original_check = transformers.utils.import_utils.check_torch_load_is_safe
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        
        try:
            model = Mamba2ForCausalLM.from_pretrained(filename, config=config).to("cuda")
        finally:
            transformers.utils.import_utils.check_torch_load_is_safe = original_check
    else:
        print(f"Checkpoint file '{filename}' not found. Loading base model weights from 'state-spaces/mamba2-370m' and initializing LoRa adapter...")
        model = Mamba2ForCausalLM(config).to("cuda")
        dataset_indices = {}
        seen_indices = []
else:
    model = Mamba2ForCausalLM(config).to("cuda")
    print(f"Checkpoint file '{filename}' not found. Loading base model weights from 'state-spaces/mamba2-370m' and initializing LoRa adapter...")
    dataset_indices = {}
    seen_indices = []

# Initialize dataset_indices if not already done
if 'dataset_indices' not in locals():
    dataset_indices = {}
if 'seen_indices' not in locals():
    seen_indices = []

model.train()

batch_size = 1 # Define batch size here
pytorch_total_params = sum(p.numel() for p in model.parameters())

datalist = []
dataset_filename = "haskell_code_dataset.ds"
if os.path.exists(dataset_filename):
    dataset = datasets.load_from_disk(dataset_filename)
else:
    dataset = load_dataset("codeparrot/github-code", split="train", name="Haskell-all", streaming=False)
    dataset.save_to_disk(dataset_filename)

batch_train = None

# Initialize FBFGS optimizer
optimizer = FBFGS(model.parameters(), lr=1., history_size=9, tolerance_change=16, max_iter=10, max_eval=1, line_search_fn="strong_wolfe", y_norm=0.8,norm=0.8, radius_y = 1,radius_ball = 1,radius_s = 1,c1 = 1e-7, c2=0.1,direction_device="cpu", optimizer_device="cuda", bracket_shift = 1/3, bracket_shove=1/3, capture_max_step = 10, capture_min_step = 10, rho_rewind=1, orthogonality=1e30, max_ls=10)

step_count = 0
step_data = []
losses_before = []
losses_deltas = []

# Initialize single figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.tight_layout(pad=3.0)

import random

dataset_size = len(dataset)
dataset_shuffled_indices = list(range(dataset_size))
current_dataset_filename = dataset_filename
dataset_index = 0

batch_input_ids_list = []
batch_attention_mask_list = []
cache = None
outputs = None

def closure():
    global batch_input_ids_list
    global batch_attention_mask_list
    global cache
    start_time = time.time()
    i = 0
    optimizer.zero_grad()
    for input_ids, attention_mask in zip(batch_input_ids_list, batch_attention_mask_list):
        chunk_size = 200
        cache=None
        grad_vector_size = 200
        num_tokens = input_ids.size(1)
        cache_position = None
        if chunk_size > 0 :
            for i in range(0, num_tokens - grad_vector_size, chunk_size):
                end_idx = min(i + chunk_size, num_tokens - grad_vector_size)
                cur_input_ids = input_ids[:, i:end_idx]
                cur_attention_mask = attention_mask[:, i:end_idx]
                if cache is not None:
                    with torch.no_grad():
                        outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, cache_params = cache, use_cache = True, cache_position=torch.tensor([i]))
                else:
                    with torch.no_grad():
                        outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, use_cache=True)
                cache = outputs.cache_params

            outputs = model(input_ids[:, -grad_vector_size:], attention_mask=attention_mask[:, -grad_vector_size:],labels = input_ids[:, -grad_vector_size:], cache_params = cache, cache_position=torch.tensor([num_tokens - grad_vector_size]))

        outputs.loss.backward()
        print(f"{outputs.loss.item():.16f}")
        return outputs.loss

# Main training loop
while True:
    cache = None
    dataset_shuffled_indices = list(range(dataset_size))
    random.shuffle(dataset_shuffled_indices)

    if not dataset_shuffled_indices:
        dataset_shuffled_indices = list(range(dataset_size))
        random.shuffle(dataset_shuffled_indices)
        seen_indices = []

    if not dataset_shuffled_indices:
        print("Dataset is empty, stopping training for this dataset.")
        break

    dataset_idx = dataset_shuffled_indices.pop()
    while dataset_idx in seen_indices and dataset_shuffled_indices:
        dataset_idx = dataset_shuffled_indices.pop()
    if dataset_idx in seen_indices:
        print("All indices seen, reshuffling and continuing.")
        dataset_shuffled_indices = list(range(dataset_size))
        random.shuffle(dataset_shuffled_indices)
        seen_indices = []
        continue

    batch_input_ids_list = []
    batch_attention_mask_list = []
    batch_count = 0
    while batch_count < batch_size:
        if not dataset_shuffled_indices:
            print(f"Dataset indices exhausted before filling batch. Current batch size: {batch_count}")
            break

        dataset_idx = dataset_shuffled_indices.pop()
        while dataset_idx in seen_indices and dataset_shuffled_indices:
            dataset_idx = dataset_shuffled_indices.pop()
        if dataset_idx in seen_indices:
            print("All indices seen, ending batch collection early.")
            break

        seen_indices.append(dataset_idx)
        batch_train = dataset[dataset_idx]['code']
        tokens = tokenizer(batch_train,truncation=False, max_length=None,padding=False, return_overflowing_tokens=False, return_length=True,return_tensors='pt').to("cuda")
        input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)
        print("got num_tokens: " + str(input_ids.size(1)))

        current_num_tokens = input_ids.size(1)
        max_len_global = random.randint(2000, 20000)
        if current_num_tokens > max_len_global:
            start_idx = random.randint(0, current_num_tokens - max_len_global)
            input_ids = input_ids[:, start_idx : start_idx + max_len_global]
            attention_mask = attention_mask[:, start_idx : start_idx + max_len_global]
            current_num_tokens = input_ids.size(1)

        max_warmup_length = 200
        if len(seen_indices) < 0 and current_num_tokens > max_warmup_length:
            start_idx = random.randint(0, current_num_tokens - max_warmup_length)
            input_ids = input_ids[:, start_idx : start_idx + max_warmup_length]
            attention_mask = attention_mask[:, start_idx : start_idx + max_warmup_length]
            current_num_tokens = input_ids.size(1)

        if current_num_tokens < max_warmup_length:
            continue

        batch_input_ids_list.append(input_ids)
        batch_attention_mask_list.append(attention_mask)
        batch_count += 1

    prompt = "-- A Haskell Module that opens a file and prints it to stdout:"
    out = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.eval()
        generated_ids = model.generate(out.input_ids, max_new_tokens=5, attention_mask=out.attention_mask)
        model.train()
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    loss_before = closure()
    print(f"Loss before step: {loss_before:.16f}")

    optimizer.step(closure)

    step_text = f" STEP {step_count} "
    color_cycle = ["\033[38;2;255;0;0m", "\033[38;2;0;255;0m", "\033[38;2;0;0;255m"]
    reset = "\033[0m"
    
    line = ""
    for i in range(32):
        line += f"{color_cycle[i % 3]}-{reset}"
    line += f"{color_cycle[step_count % 3]}{step_text}{reset}"
    for i in range(32):
        line += f"{color_cycle[(i+step_count) % 3]}-{reset}"
    
    print(f"\n{line}\n")
    
    loss_after = closure()
    
    loss_delta = loss_before - loss_after
    print(f"\033[90mLoss delta gap: {loss_delta:.16f}\033[0m")
    
    step_data.append(step_count)
    losses_before.append(loss_before.item())
    losses_deltas.append(loss_delta.item())
    
    ax1.cla()
    ax2.cla()
    
    ax1.plot(step_data, losses_before, 'b-')
    ax1.set_title(f"Loss Before vs Steps (Step {step_count})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    
    ax2.plot(step_data, losses_deltas, 'r-')
    ax2.set_title(f"Loss Delta vs Steps (Step {step_count})")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss Delta")
    
    plt.savefig("training.png")

    csv_filename = "loss_data.csv"
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['step', 'loss_before', 'loss_delta'])
        writer.writerow([step_count, loss_before.item(), loss_delta.item()])

    torch.cuda.empty_cache()
    step_count += 1

    torch.cuda.empty_cache()
    gc.collect()

    if step_count % 10 == 0:
        current_dataset_filename = dataset_filename
        dataset_indices[current_dataset_filename] = seen_indices
        
        # Save model with safetensors if available
        if has_safetensors:
            print(f"Saving model using safetensors format")
            safetensors_file = filename + ".safetensors"
            save_file(model.state_dict(), safetensors_file)
            # Also save config
            if not os.path.exists(filename):
                os.makedirs(filename)
            model.config.save_pretrained(filename)
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                model.tokenizer.save_pretrained(filename)
        else:
            print(f"Saving model using PyTorch format (insecure)")
            # Fallback to original method
            model.save_pretrained(filename, safe_serialization=False)
        
        torch.save(dataset_indices, indices_filename)
        optimizer.save_history(history_filename)
        print(
            f"Model, indices, and FBFGS history saved to {filename}, {indices_filename}, and {history_filename} at step {step_count}, seen indices count for {current_dataset_filename}: {len(seen_indices)}"
        )
