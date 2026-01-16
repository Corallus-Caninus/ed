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


num_cores = os.cpu_count()
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

filename = "AI_Checkpoint.ai"
history_filename = "fbfgs_history.pth"
indices_filename = "dataset_indices.pth"

# Load GPT-2 tokenizer (byte-level BPE tokenizer) with maximum sequence length
print("Loading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set pad token to eos token for compatibility
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Set maximum sequence length to a billion (effectively no truncation)
tokenizer.model_max_length = 1000000000  # 1 billion tokens
tokenizer.truncation_side = "right"  # Truncate from the right (preserves beginning)
tokenizer.padding_side = "right"  # Pad from the right

print(f"Successfully loaded GPT-2 tokenizer with vocab size: {tokenizer.vocab_size} and max length: {tokenizer.model_max_length}")

# Load the base config from the 370m model and modify specific parameters
config = Mamba2Config.from_pretrained("state-spaces/mamba2-370m")
# Reduced parameter configuration (~20M parameters)
config.hidden_size = 128      # Reduced from 200 (primary reduction)
config.num_hidden_layers = 20 # Reduced from 24
config.num_heads = 16          # Kept same (1216/16 = 16 head_dim)
config.head_dim = 16          # Must be hidden_size/num_heads = 128/8 = 16
config.state_size = 16        # Reduced from 54 (SSM state size)
config.dtype = "float16"

# Set the model's maximum sequence length to a billion (effectively no truncation)
config.seq_len = 1000000000  # 1 billion tokens

# Update config vocab_size to match tokenizer
config.vocab_size = tokenizer.vocab_size
print(f"Updated config vocab_size to: {config.vocab_size}")
print(f"Model seq_len set to: {config.seq_len}")

# Always use custom config for both saving and loading
# Try to use GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if os.path.exists(filename):
    print(f"Loading model from checkpoint: {filename}")
    # Load model with the updated config (including vocab_size)
    # If vocab_size differs from checkpoint, resize embeddings automatically
    try:
        model = Mamba2ForCausalLM.from_pretrained(filename, config=config).to(device)
        print(f"Model loaded successfully with vocab_size: {config.vocab_size}")
    except Exception as e:
        print(f"Failed to load model with new config: {e}")
        print("Trying to load and resize embeddings...")
        model = Mamba2ForCausalLM.from_pretrained(filename).to(device)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Model loaded and embeddings resized to match tokenizer vocab size: {len(tokenizer)}")
else:
    print(f"Checkpoint not found. Creating new model with custom config...")
    model = Mamba2ForCausalLM(config).to(device)
    dataset_indices = {}
    seen_indices = []

# Initialize dataset_indices if not already done
if 'dataset_indices' not in locals():
    dataset_indices = {}
if 'seen_indices' not in locals():
    seen_indices = []

model.train()

# Ensure all parameters have gradients enabled and verify
params_without_gradients = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        params_without_gradients.append(name)
        param.requires_grad = True  # Enable gradients if disabled

if params_without_gradients:
    print(f"WARNING: Found {len(params_without_gradients)} parameters without gradients enabled. Enabled gradients for:")
    for name in params_without_gradients:
        print(f"  - {name}")
else:
    print("All parameters have gradients enabled.")

# Verify all parameters have gradients after setting
all_have_gradients = True
for name, param in model.named_parameters():
    if not param.requires_grad:
        all_have_gradients = False
        break

if not all_have_gradients:
    raise AssertionError("Some parameters still don't have gradients enabled after initialization!")

batch_size = 1 # Define batch size here (changed from 1 to 1)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {pytorch_total_params:,} parameters")

datalist = []
dataset_filename = "haskell_code_dataset.ds"
if os.path.exists(dataset_filename):
    dataset = datasets.load_from_disk(dataset_filename)
else:
    dataset = load_dataset("codeparrot/github-code", split="train", name="Haskell-all", streaming=False)
    dataset.save_to_disk(dataset_filename)

batch_train = None

# Initialize FBFGS optimizer
optimizer_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using optimizer device: {optimizer_device}")
optimizer = FBFGS(model.parameters(), lr=1., history_size=9, tolerance_change=16, max_iter=10, max_eval=1, line_search_fn="strong_wolfe", y_norm=1.1, norm=1, radius_y = 1e-8,radius_ball = 1,radius_s = 1e-20,c1 = 1e-7, c2=0.1,direction_device="cpu", optimizer_device=optimizer_device, bracket_shift = 1/3, bracket_shove=1/3, capture_max_step = 10, capture_min_step = 0.01, rho_rewind=10, orthogonality=1, max_ls=10)

# Load FBFGS history if it exists
if os.path.exists(history_filename):
    # Allow the SparseFlatTensor class from fbfgs module for safe loading
    import torch.serialization
    try:
        from fbfgs.fbfgs import SparseFlatTensor
        torch.serialization.add_safe_globals([SparseFlatTensor])
    except ImportError:
        # If SparseFlatTensor can't be imported, we'll try loading anyway
        pass
    
    # Try to load with weights_only=False to handle custom classes
    try:
        optimizer.load_history(history_filename)
        print(f"Loaded FBFGS history from {history_filename}")
    except Exception as e:
        print(f"Error loading FBFGS history: {e}. Starting from scratch.")

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
# Global list to store random starts for each sequence in the current batch
random_starts_list = []
# Global list to store sequence indices for the current batch
sequence_indices_list = []

def closure():
    global batch_input_ids_list
    global batch_attention_mask_list
    global random_starts_list
    start_time = time.time()
    optimizer.zero_grad()
    
    # Process all sequences in the batch sequentially
    losses_random = []
    losses_last = []
    num_random_positions = 5  # Default number of random positions to generate gradients at
    
    for seq_idx, (input_ids, attention_mask) in enumerate(zip(batch_input_ids_list, batch_attention_mask_list)):
        grad_vector_size = 2  # Changed from 20 to 2
        num_tokens = input_ids.size(1)
        chunk_size = 1500
        
        # Get the pre-calculated random start positions for this sequence
        # Now we have multiple random starts
        random_starts = random_starts_list[seq_idx]  # This is now a list of start positions
        
        # Sort random starts to ensure proper processing order
        random_starts = sorted(random_starts)
        
        # Build cache up to the first random start (without gradients)
        cache = None
        if chunk_size > 0 and num_tokens > 0:
            # First, process up to first random start without gradients (if any)
            first_random_start = random_starts[0] if random_starts else 0
            for i in range(0, first_random_start, chunk_size):
                end_idx = min(i + chunk_size, first_random_start)
                if end_idx > i:  # Only process if there's something
                    cur_input_ids = input_ids[:, i:end_idx]
                    cur_attention_mask = attention_mask[:, i:end_idx]
                    with torch.no_grad():
                        if cache is not None:
                            outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, cache_params = cache, use_cache = True, cache_position=torch.tensor([i]))
                        else:
                            outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, use_cache=True)
                        cache = outputs.cache_params
            
            # Process each random segment with gradients
            for random_start in random_starts:
                # Process up to this random start from the previous position (if there's a gap)
                if random_start > first_random_start:
                    # Check if there's a gap between previous position and current random start
                    # We need to track the last position we processed
                    # For simplicity, we'll process from the previous random start + grad_vector_size
                    prev_end = random_start  # We'll process up to this point without gradients
                    for i in range(0, prev_end, chunk_size):
                        end_idx = min(i + chunk_size, prev_end)
                        if end_idx > i and cache is not None:  # Only process if there's something and we have cache
                            cur_input_ids = input_ids[:, i:end_idx]
                            cur_attention_mask = attention_mask[:, i:end_idx]
                            with torch.no_grad():
                                outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, cache_params = cache, use_cache = True, cache_position=torch.tensor([i]))
                                cache = outputs.cache_params
                
                # Now process the random segment with gradients
                random_input_ids = input_ids[:, random_start:random_start+grad_vector_size]
                random_attention_mask = attention_mask[:, random_start:random_start+grad_vector_size]
                random_outputs = model(input_ids=random_input_ids, attention_mask=random_attention_mask, labels=random_input_ids, cache_params=cache, cache_position=torch.tensor([random_start]))
                loss_random = random_outputs.loss
                losses_random.append(loss_random)
                
                # IMPORTANT: Reuse the cache from gradient computation instead of recomputing
                # This cache already contains information from random_start position
                cache = random_outputs.cache_params
                
                # Move to position after this random segment
                first_random_start = random_start + grad_vector_size
            
            # After processing all random segments, process up to the last segment
            # Process from the end of the last random segment to num_tokens-grad_vector_size without gradients
            for i in range(first_random_start, num_tokens - grad_vector_size, chunk_size):
                end_idx = min(i + chunk_size, num_tokens - grad_vector_size)
                if end_idx > i and cache is not None:  # Only process if there's something and we have cache
                    cur_input_ids = input_ids[:, i:end_idx]
                    cur_attention_mask = attention_mask[:, i:end_idx]
                    with torch.no_grad():
                        outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, cache_params = cache, use_cache = True, cache_position=torch.tensor([i]))
                        cache = outputs.cache_params
            
            # Now process last segment with gradients (grad_vector_size tokens)
            last_input_ids = input_ids[:, -grad_vector_size:]
            last_attention_mask = attention_mask[:, -grad_vector_size:]
            last_outputs = model(input_ids=last_input_ids, attention_mask=last_attention_mask, labels=last_input_ids, cache_params=cache, cache_position=torch.tensor([num_tokens - grad_vector_size]))
            loss_last = last_outputs.loss
            losses_last.append(loss_last)
            
            # Average the losses for this sequence and perform backward pass
            # If we have multiple random segments, average them
            if random_starts:
                random_loss_avg = torch.mean(torch.stack(losses_random[-len(random_starts):]))
                seq_loss = (random_loss_avg + loss_last) / 2
            else:
                seq_loss = loss_last
            seq_loss.backward()
        else:
            # Fallback to old method if chunk_size is 0 or no tokens
            if cache is not None:
                outputs = model(input_ids=input_ids, attention_mask = attention_mask, labels = input_ids, cache_params = cache, use_cache = True, cache_position=torch.tensor([0]))
            else:
                outputs = model(input_ids=input_ids, attention_mask = attention_mask, labels = input_ids, use_cache=True)
            loss_random = outputs.loss
            loss_last = outputs.loss
            losses_random.append(loss_random)
            losses_last.append(loss_last)
            # Average the losses for this sequence and perform backward pass
            seq_loss = (loss_random + loss_last) / 2
            seq_loss.backward()
    
    # Calculate average losses for logging
    loss_random_avg = torch.mean(torch.stack(losses_random)) if losses_random else torch.tensor(0.0)
    loss_last_avg = torch.mean(torch.stack(losses_last)) if losses_last else torch.tensor(0.0)
    loss_avg = (loss_random_avg + loss_last_avg) / 2
    
    print(f"Random segment loss: {loss_random_avg.item():.16f}")
    print(f"Last segment loss: {loss_last_avg.item():.16f}")
    print(f"Averaged loss: {loss_avg.item():.16f}")
    return loss_avg

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
        # Use tokenizer with maximum length (1 billion tokens)
        tokens = tokenizer(
            batch_train,
            truncation=False,  # Disable truncation
            max_length=1000000000,  # 1 billion tokens
            padding=False,
            return_overflowing_tokens=False,
            return_length=True,
            return_tensors='pt'
        ).to(device)
        input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)
        
        # Verify token IDs are valid (should always be true with byte-level tokenizer)
        # This assertion ensures we don't have invalid tokens
        if input_ids.max().item() >= tokenizer.vocab_size:
            raise ValueError(f"Token ID >= vocab_size: {input_ids.max().item()} >= {tokenizer.vocab_size}. This should not happen with byte-level tokenizer.")
        
        print(f"Got num_tokens: {input_ids.size(1)}")
        
        # Removed truncation logic as requested

        current_num_tokens = input_ids.size(1)
        
        # Apply some length filtering/truncation based on your training preferences
        # This is optional - adjust as needed
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

    # Calculate random starts for each sequence in the batch BEFORE calling closure
    grad_vector_size = 2  # Must match the grad_vector_size in closure
    random_starts_list = []
    sequence_indices_list = []
    
    for seq_idx, (input_ids, attention_mask) in enumerate(zip(batch_input_ids_list, batch_attention_mask_list)):
        num_tokens = input_ids.size(1)
        
        # Get multiple random start positions for the random segments
        # We'll generate 5 random positions (default)
        num_random_positions = 5  # Default number of random positions
        
        # Ensure we have enough tokens for all random positions plus the last segment
        required_tokens = (num_random_positions + 1) * grad_vector_size
        
        if num_tokens < required_tokens:
            # Not enough tokens for all random positions, use fewer or skip
            # For now, we'll use what we can
            num_random_positions = max(0, (num_tokens // grad_vector_size) - 1)
        
        random_starts = []
        if num_random_positions > 0:
            # Generate random positions ensuring they don't overlap and leave room for last segment
            # We'll generate positions that are at least grad_vector_size apart
            available_positions = num_tokens - required_tokens + grad_vector_size  # Adjust for random starts
            
            if available_positions > 0:
                # Generate unique random positions
                potential_starts = list(range(0, available_positions, grad_vector_size))
                if len(potential_starts) >= num_random_positions:
                    random_starts = random.sample(potential_starts, num_random_positions)
                else:
                    # Use all available positions
                    random_starts = potential_starts
            else:
                # Not enough space, use a single position at the beginning
                random_starts = [0]
        
        random_starts_list.append(random_starts)
        sequence_indices_list.append(seq_idx)
    
    print(f"Calculated random starts for batch: {[rs if isinstance(rs, list) else [rs] for rs in random_starts_list]}")

    prompt = "-- A Haskell Module that opens a file and prints it to stdout:"
    out = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Verify generation prompt token IDs are valid
    if out.input_ids.max().item() >= tokenizer.vocab_size:
        raise ValueError(f"Prompt token ID >= vocab_size: {out.input_ids.max().item()} >= {tokenizer.vocab_size}")
    with torch.no_grad():
        model.eval()
        try:
            generated_ids = model.generate(out.input_ids, max_new_tokens=5, attention_mask=out.attention_mask)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Generated text: {generated_text}")
        except Exception as e:
            print(f"Generation failed: {e}")
            generated_text = "Generation failed"
        model.train()

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

    step_count += 1
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if device == "cuda":
        torch.cuda.empty_cache()

    if step_count % 10 == 0:
        current_dataset_filename = dataset_filename
        dataset_indices[current_dataset_filename] = seen_indices
        
        # Save model using PyTorch format
        print(f"Saving model...")
        model.save_pretrained(filename, safe_serialization=False)
        
        tokenizer.save_pretrained(filename)
        
        torch.save(dataset_indices, indices_filename)
        optimizer.save_history(history_filename)
        print(
            f"Model, indices, and FBFGS history saved to {filename}, {indices_filename}, and {history_filename} at step {step_count}, seen indices count for {current_dataset_filename}: {len(seen_indices)}"
        )
