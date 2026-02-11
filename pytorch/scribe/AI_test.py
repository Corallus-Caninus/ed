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
config = Mamba2Config.from_pretrained("AntonV/mamba2-370m-hf")
# Reduced parameter configuration (~20M parameters)
config.hidden_size = 200      # Reduced from 200 (primary reduction)
config.num_hidden_layers = 60 # Reduced from 24
config.num_heads = 5          # Kept same (125/5 = 5 head_dim)
config.head_dim = 80          # Must be hidden_size/num_heads = 128/8 = 80
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
optimizer = FBFGS(model.parameters(),  history_size=9, tolerance_change=0.01, max_iter=10,  line_search_fn="strong_wolfe", y_norm=1.5, norm=1.33, radius_y=5e4, radius_ball=500, radius_ball_s=500, radius_s=1e6, c1=0, c2=0.1, direction_device="cpu", optimizer_device=optimizer_device, bracket_shift=1/3, bracket_shove=1/3, capture_max_step=10, capture_min_step=0.001, rho_rewind=3, orthogonality=0.001, max_ls=5, norm_group_s=5, norm_group_y=0.2, prefetch_buffer=50e6)# TODO: try reducing tolerance change with angle based orthogonality since it doesnt converge the direction now (more point breaks)
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
loss_without_regularizer = 0.0  # Track pure loss for graphing
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
# Global variable to store the current batch random starts (calculated once per step)
current_batch_random_starts = []
def closure():
    global batch_input_ids_list
    global batch_attention_mask_list
    start_time = time.time()
    optimizer.zero_grad()
    
    # We'll accumulate loss values for averaging and call backward immediately
    total_loss_sum = 0.0
    loss_count = 0
    
    if not batch_input_ids_list:
        # No valid sequences in batch
        dummy_loss = torch.tensor(0.0, requires_grad=True).to(device)
        dummy_loss.backward()
        print(f"Averaged loss: {dummy_loss.item():.16f}")
    
    grad_vector_size = 2
    chunk_size = 10000
    
    # Process each sequence in the batch separately (no padding)
    # Use the pre-calculated random starts (calculated once per step)
    print("Random starts for each sequence in batch:")
    
    for seq_idx in range(len(batch_input_ids_list)):
        input_ids = batch_input_ids_list[seq_idx]
        attention_mask = batch_attention_mask_list[seq_idx]
        
        num_tokens = input_ids.size(1)
        
        # Use pre-calculated random starts for this sequence
        if seq_idx < len(current_batch_random_starts):
            random_starts = current_batch_random_starts[seq_idx]
        else:
            # Fallback if for some reason we don't have pre-calculated starts
            random_starts = []
            num_random_positions = 1
            required_tokens = (num_random_positions + 1) * grad_vector_size
            
            if num_tokens >= required_tokens:
                available_positions = num_tokens - required_tokens + grad_vector_size
                if available_positions > 0:
                    potential_starts = list(range(0, available_positions, grad_vector_size))
                    if len(potential_starts) >= num_random_positions:
                        random_starts = random.sample(potential_starts, num_random_positions)
                    else:
                        random_starts = potential_starts
        
        # Print the random starts for this sequence
        print(f"  Sequence {seq_idx}: {random_starts} (sequence length: {num_tokens})")
        
        # Process this sequence
        cache = None
        
        if chunk_size > 0 and num_tokens > 0:
            # Build cache up to the first random start (without gradients)
            first_random_start = random_starts[0] if random_starts else 0
            for i in range(0, first_random_start, chunk_size):
                end_idx = min(i + chunk_size, first_random_start)
                if end_idx > i:
                    cur_input_ids = input_ids[:, i:end_idx]
                    cur_attention_mask = attention_mask[:, i:end_idx]
                    with torch.no_grad():
                        if cache is not None:
                            outputs = model(input_ids=cur_input_ids, attention_mask=cur_attention_mask, 
                                           labels=cur_input_ids, cache_params=cache, use_cache=True, 
                                           cache_position=torch.tensor([i]))
                        else:
                            outputs = model(input_ids=cur_input_ids, attention_mask=cur_attention_mask, 
                                           labels=cur_input_ids, use_cache=True)
                        cache = outputs.cache_params
            
            # Process each random segment for this sequence
            for random_start in random_starts:
                # Process up to this random start from the previous position (if there's a gap)
                if random_start > first_random_start:
                    prev_end = random_start
                    for i in range(0, prev_end, chunk_size):
                        end_idx = min(i + chunk_size, prev_end)
                        if end_idx > i and cache is not None:
                            cur_input_ids = input_ids[:, i:end_idx]
                            cur_attention_mask = attention_mask[:, i:end_idx]
                            with torch.no_grad():
                                outputs = model(input_ids=cur_input_ids, attention_mask=cur_attention_mask, 
                                               labels=cur_input_ids, cache_params=cache, use_cache=True, 
                                               cache_position=torch.tensor([i]))
                                cache = outputs.cache_params
                
                # Now process the random segment and call backward immediately
                random_input_ids = input_ids[:, random_start:random_start+grad_vector_size]
                random_attention_mask = attention_mask[:, random_start:random_start+grad_vector_size]
                random_outputs = model(input_ids=random_input_ids, attention_mask=random_attention_mask, 
                                     labels=random_input_ids, cache_params=cache, 
                                     cache_position=torch.tensor([random_start]))
                loss_random = random_outputs.loss
                # Call backward immediately for this loss
                loss_random.backward()
                total_loss_sum += loss_random.item()
                loss_count += 1
                
                # IMPORTANT: Reuse the cache from gradient computation instead of recomputing
                cache = random_outputs.cache_params
                
                # Move to position after this random segment
                first_random_start = random_start + grad_vector_size
            
            # After processing all random segments, process up to the last segment
            for i in range(first_random_start, num_tokens - grad_vector_size, chunk_size):
                end_idx = min(i + chunk_size, num_tokens - grad_vector_size)
                if end_idx > i and cache is not None:
                    cur_input_ids = input_ids[:, i:end_idx]
                    cur_attention_mask = attention_mask[:, i:end_idx]
                    with torch.no_grad():
                        outputs = model(input_ids=cur_input_ids, attention_mask=cur_attention_mask, 
                                       labels=cur_input_ids, cache_params=cache, use_cache=True, 
                                       cache_position=torch.tensor([i]))
                        cache = outputs.cache_params
            
            # Now process last segment for this sequence and call backward immediately
            last_input_ids = input_ids[:, -grad_vector_size:]
            last_attention_mask = attention_mask[:, -grad_vector_size:]
            last_outputs = model(input_ids=last_input_ids, attention_mask=last_attention_mask, 
                               labels=last_input_ids, cache_params=cache, 
                               cache_position=torch.tensor([num_tokens - grad_vector_size]))
            loss_last = last_outputs.loss
            # Call backward immediately for this loss
            loss_last.backward()
            total_loss_sum += loss_last.item()
            loss_count += 1
            
        else:
            # Fallback to old method if chunk_size is 0 or no tokens
            if cache is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, 
                              cache_params=cache, use_cache=True, cache_position=torch.tensor([0]))
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=True)
            loss_random = outputs.loss
            loss_last = outputs.loss
            # Call backward immediately for both losses
            loss_random.backward()
            loss_last.backward()
            total_loss_sum += loss_random.item()
            total_loss_sum += loss_last.item()
            loss_count += 2
    
    # Calculate total loss as average of all collected losses
    if loss_count > 0:
        total_loss = total_loss_sum / loss_count
    else:
        # If we have no losses at all, we need to handle this case
        dummy_loss = torch.tensor(0.0, requires_grad=True).to(batch_input_ids_list[0].device)
        dummy_loss.backward()
        total_loss = dummy_loss.item()
    
    print(f"Averaged loss: {total_loss:.16f}")
    total_loss_tensor = torch.tensor(total_loss, requires_grad=True).to(batch_input_ids_list[0].device)
    global loss_without_regularizer
    loss_without_regularizer = total_loss  # Store pure loss for logging
    # Add regularization term as dot product of gradients and parameters
    reg_term = torch.zeros(1, requires_grad=True).to(batch_input_ids_list[0].device)
    reg_count = 0
    for name, param in model.named_parameters():
        pdp = torch.sqrt(torch.dot(param.view(-1), param.view(-1)))
        pdg = torch.dot(param.grad.view(-1), param.view(-1))
        if param is not None   and pdp > 50 :
##            reg_term += torch.sum(param.grad * param.data).item()
#Lambda set to the cosine_similarity of grad on param to prevent gradient from being dominated by the param decay while maximizing decay
#If its already reducing (negative p@g) than let it decay by the data instead of bleeding it
            if pdg > 0:
                lam = pdg/ ((pdp-50) * torch.sqrt(torch.dot(param.grad.view(-1), param.grad.view(-1))))
                param.grad += param*lam
                print("Triggered event horizon.."+ " PDP: " + str(pdp) + " lam: " + str(lam))
            if pdg == 0:
                lam = torch.sqrt(torch.dot(param.grad.view(-1), param.grad.view(-1)))
                param.grad += param*lam
                print("Triggered orthogonal event horizon.."+ " PDP: " + str(pdp) + " lam: " + str(lam))
            if torch.dot(param.grad.view(-1), param.grad.view(-1)) == 0:
                param.grad += param*pdp - 50
# TODO: handle orthogonality with magnitude (0.5* grad@grad)
#            gdg = torch.dot(param.grad.view(-1), param.grad.view(-1))
#            pdg = torch.dot(param.grad.view(-1), param.view(-1))
#            if pdg.item() > 0:
## TODO: params magnitude arent in this equation, ensure we dont blow up the logits
## TODO: after this blows up, try increasing the regularizer aggressively since it seems we blow up the logits first then overfit the regularizer. If we never blow up the logits we fix the source of the problem.
##                reg_term = reg_term + (pdg/torch.sqrt(gdg)) * pdp
#                reg_term = reg_term + pdg
#                reg_count += 1
##                pdp = torch.dot(param.view(-1), param.view(-1))
##                l2_decay = torch.sqrt(pdp)
##                reg_term = reg_term + pdp*(2/(1+2.7**(-2.7*l2_decay)/500) - 1)
###                cosine_similarity = pdg/ (torch.sqrt(torch.dot(param.view(-1), param.view(-1)))* torch.sqrt(torch.dot(param.grad.view(-1), param.grad.view(-1))))
###                reg_delta =  cosine_similarity
#### TODO always True
###                if reg_delta > 0:
###                    composite_loss = reg_term + reg_delta
###                    reg_count += 1
#### TODO: TEST ME. NOTE: this is a false positive for negative orthogonality but we want GSO to hit warp drive on reduction
#            if pdg.item() == 0:
### TODO: pytorch sigmoid is surely faster
### TODO: 0.5?
##                reg_term = reg_term +  torch.dot(param.grad.view(-1), param.grad.view(-1))**2
#                reg_term = reg_term+ torch.sqrt(torch.dot(param.grad.view(-1), param.grad.view(-1)) *  1/(1+e**(-torch.dot(param.grad.view(-1), param.grad.view(-1)))))
#                reg_count += 1
#### TODO always True
###                if reg_delta > 0:
###                    reg_term = reg_term + reg_delta
###                    reg_count += 1
#                print("hit ortho")
###                reg_term = reg_term + torch.sqrt(torch.dot(param.grad.view(-1), param.grad.view(-1)).item())
## TODO: orthogonal addition after event horizon regularizer
#    # Create composite loss
## NOTE: We perform the product here to resist the strong regularizer from overtaking the objective function
#    print("reg term: " + str(reg_term))
#    # Perform second backward pass on composite loss
#    if reg_term > 0:
#        reg_term =  min(total_loss_tensor, reg_term)
#        reg_term.backward()
##    reg_term.backward()
#    print(f"Composite loss: " + str(reg_term))
##TODO: only graph the loss function not the regularizer too
    return total_loss_tensor.item() #reg_term.item()+ total_loss_tensor.item()
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
    # Collect batch samples as code strings first
    batch_samples = []
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
        
        # Apply length filtering/truncation
        # First tokenize to get length
        test_tokens = tokenizer(
            batch_train,
            truncation=False,
            max_length=1000000000,
            padding=False,
            return_length=True,
            return_tensors='pt'
        ).to(device)
        
        current_num_tokens = test_tokens.input_ids.size(1)
        
        # Apply length filtering
        max_len_global = random.randint(2000, 20000)
        if current_num_tokens > max_len_global:
            start_idx = random.randint(0, current_num_tokens - max_len_global)
            # Tokenize again with truncation
            truncated_tokens = tokenizer(
                batch_train,
                truncation=False,
                max_length=1000000000,
                padding=False,
                return_tensors='pt'
            ).to(device)
            input_ids = truncated_tokens.input_ids[:, start_idx : start_idx + max_len_global]
            attention_mask = truncated_tokens.attention_mask[:, start_idx : start_idx + max_len_global]
            current_num_tokens = input_ids.size(1)
        
#        max_warmup_length = 200
#        if len(seen_indices) < 0 and current_num_tokens > max_warmup_length:
#            start_idx = random.randint(0, current_num_tokens - max_warmup_length)
#            # Tokenize again with truncation
#            truncated_tokens = tokenizer(
#                batch_train,
#                truncation=False,
#                max_length=1000000000,
#                padding=False,
#                return_tensors='pt'
#            ).to(device)
#            input_ids = truncated_tokens.input_ids[:, start_idx : start_idx + max_warmup_length]
#            attention_mask = truncated_tokens.attention_mask[:, start_idx : start_idx + max_warmup_length]
#            current_num_tokens = input_ids.size(1)
        
# TODO: we do this for testing but on a production train we want all the tokens
#        if  current_num_tokens > max_warmup_length:
        if  current_num_tokens > 10000:
            continue
        
        # Add the processed code to batch samples
        batch_samples.append(batch_train)
        batch_count += 1
    
    # Now tokenize all samples in the batch without padding
    if batch_samples:
        # Tokenize each sequence separately to avoid padding
        batch_input_ids_list = []
        batch_attention_mask_list = []
        
        for sample in batch_samples:
            tokens = tokenizer(
                sample,
                truncation=False,  # Disable truncation (we already truncated if needed)
                max_length=1000000000,  # 1 billion tokens
                padding=False,  # No padding - we'll process each sequence separately
                return_overflowing_tokens=False,
                return_length=True,
                return_tensors='pt'
            ).to(device)
            
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
            
            # Verify token IDs are valid
            if input_ids.max().item() >= tokenizer.vocab_size:
                raise ValueError(f"Token ID >= vocab_size: {input_ids.max().item()} >= {tokenizer.vocab_size}. This should not happen with byte-level tokenizer.")
            
            # Add to our lists (each sequence is processed separately)
            batch_input_ids_list.append(input_ids)
            batch_attention_mask_list.append(attention_mask)
        
        print(f"Processed batch with {len(batch_input_ids_list)} sequences")
    else:
        # No valid samples found, skip this batch
        batch_input_ids_list = []
        batch_attention_mask_list = []
    
    # Calculate random starts once per step (same for both closure calls in this step)
    grad_vector_size = 2
    num_random_positions = 1
    
    current_batch_random_starts = []
    
    if batch_input_ids_list:
        print("Calculating random starts for the current batch (will be same for both closure calls in this step):")
        for seq_idx, input_ids in enumerate(batch_input_ids_list):
            num_tokens = input_ids.size(1)
            required_tokens = (num_random_positions + 1) * grad_vector_size
            random_starts = []
            
            if num_tokens >= required_tokens:
                available_positions = num_tokens - required_tokens + grad_vector_size
                if available_positions > 0:
                    potential_starts = list(range(0, available_positions, grad_vector_size))
                    if len(potential_starts) >= num_random_positions:
                        random_starts = random.sample(potential_starts, num_random_positions)
                    else:
                        random_starts = potential_starts
            
            current_batch_random_starts.append(random_starts)
            print(f"  Sequence {seq_idx}: {random_starts} (sequence length: {num_tokens})")
    else:
        current_batch_random_starts = []
    
    # Remove the random_starts_list calculation - it's now done once per step
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
    loss_before = loss_without_regularizer
    print(f"Loss before step: {loss_without_regularizer:.16f}")
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
    loss_after = loss_without_regularizer
    
    loss_delta = loss_before - loss_after  # Use pure loss before - pure loss after
#TODO: reset params here if gap is negative as a test
    print(f"\033[90mLoss delta gap: {loss_delta:.16f}\033[0m")
    
    step_data.append(step_count)
    losses_before.append(loss_before)
    losses_deltas.append(loss_delta)
    
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
        writer.writerow([step_count, loss_before, loss_delta])
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
