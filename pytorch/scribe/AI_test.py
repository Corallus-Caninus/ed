import os
import math
import sys
import csv
import matplotlib.pyplot as plt

import torch

#print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

import gc
from transformers import MambaConfig, Mamba2ForCausalLM, AutoTokenizer,  AutoModelForCausalLM, AutoConfig, Mamba2Config

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from fbfgs import FBFGS
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import datasets
from datasets import Dataset
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import LoraConfig, get_peft_model, BoneConfig, BoneModel, LoraModel
from peft import PeftModel
torch.backends.cudnn.enabled = False
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import os
num_cores = os.cpu_count()
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

accelerator = Accelerator()
filename = "AI_Checkpoint.ai"
#TODO: project Basilisk: parallelize the model layer-wise with the gradients. Also parallelize the flat-grads and gtd etc in L-BFGS-N. Simplest parallelization, assuming we are using commodity last-gen accelerators for edge learning, this will allow the most performant scale-out of models (e.g.: 3 k80's or 3 MI25's)

import time
#INIT START
model_id = "mistralai/Mamba-Codestral-7B-v0.1"
dataset_filename = "haskell_code_dataset.ds"
model_id = "hanzla/Falcon3-Mamba-R1-v0"
model_id = "state-spaces/mamba2-370m"
model_id = "AntonV/mamba2-370m-hf"
history_filename = "fbfgs_history.pth"
indices_filename = "dataset_indices.pth"

# Load the base config from the 370m model and modify specific parameters
config = Mamba2Config.from_pretrained(model_id, trust_remote_code=True)
config.hidden_size = 200     
config.num_hidden_layers = 24   
config.head_dim = 50
config.num_heads = 8
config.state_size = 54
config.dtype= torch.float16

#model = Mamba2ForCausalLM(config)

# Load tokenizer from the initialized model instead of the pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

if os.path.exists(filename): # Load model weights and optimizer history
    #print(f"Checkpoint file '{filename}' found. Loading LoRa adapter from checkpoint...")
    model = Mamba2ForCausalLM.from_pretrained(filename, config=config).to("cuda")
    dataset_indices = {}

    # Print requires_grad status *before* dtype conversion
    #print("--- Parameter requires_grad status (after PeftModel.from_pretrained) ---")
    #for name, param in model.named_parameters():
        #if "lora_" in name or param.requires_grad: # Print Lora params or any trainable param
             #print(f"  Param: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    #print("--- End Parameter requires_grad status ---")

    current_dataset_filename = dataset_filename # Define current dataset filename
    if os.path.exists(indices_filename):
        dataset_indices = torch.load(indices_filename)
        print("After loading - dataset_indices:", dataset_indices)
        seen_indices = dataset_indices.get(current_dataset_filename, [])
        #print(f"Model checkpoint loaded successfully from '{filename}'. Resuming {current_dataset_filename} with {len(seen_indices)} indices seen.")
        #if dataset_indices:
            #print("Warning: Checkpoint contains dataset indices, ensure you are using the correct dataset or intend to resume.")
    else: # This else belongs to the inner if
        dataset_indices = {} # Initialize dataset_indices for new run
        seen_indices = [] # Initialize seen_indices for new run
        #print(f"Model checkpoint loaded successfully from '{filename}'. Starting new run for {current_dataset_filename}.") # Print message for new run

else:
    model=Mamba2ForCausalLM(config).to("cuda")
    print(f"Checkpoint file '{filename}' not found. Loading base model weights from '{model_id}' and initializing LoRa adapter...")
#    config = Mamba2Config.from_pretrained(model_id, trust_remote_code=True)
#    model = Mamba2ForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.float16, trust_remote_code=True, device_map="cpu")
    #print("--- Model Named Parameters (freshly loaded base model) ---")
    #for name, param in model.named_parameters(): # Non-recursive for brevity initially
        #print(f"Parameter Name: {name}, Parameter Shape: {param.shape}")
    #print("--- End Model Inspection (freshly loaded base model) ---")
    dataset_indices = {}
    current_dataset_filename = dataset_filename # Define current dataset filename
    seen_indices = [] # Initialize seen_indices for new run
#model.gradient_checkpointing_enable()
model.train()

batch_size = 1 # Define batch size here
pytorch_total_params = sum(p.numel() for p in model.parameters())

#print("num parameters: " + str(pytorch_total_params))

datalist = []
if os.path.exists(dataset_filename):
    if os.path.exists(dataset_filename):
      dataset = datasets.load_from_disk(dataset_filename)
else:
    #dataset = load_dataset("kye/all-torvalds-c-code-1", split="train", name="default")
    dataset = load_dataset("codeparrot/github-code", split="train", name="Haskell-all", streaming=False)
    #dataset = load_dataset("codeparrot/github-code", split="train", name="C-all",streaming=True)
    dataset.save_to_disk(dataset_filename)

batch_train = None

# Initialize Adam optimizer
optimizer = FBFGS(model.parameters(), lr=1., history_size=9, tolerance_change=16, max_iter=10, max_eval=1, line_search_fn="strong_wolfe", y_norm=1.2, norm=1.2, radius_y = 50,radius_ball = 10,radius_s=10,c1 = 1e-7, c2=0.7,direction_device="cpu", optimizer_device="cuda", bracket_shift = 1/3, bracket_shove=1/3, capture_max_step = 10, capture_min_step = 0.01, rho_rewind=1, orthogonality=0.5, max_ls=10)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# if os.path.exists(history_filename):  #Load optimizer history if checkpoint exists
    # optimizer.load_history(history_filename)
#INIT END

step_count = 0
step_data = []
losses_before = []
losses_deltas = []

# Initialize single figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.tight_layout(pad=3.0)

import random

dataset_size = len(dataset)
dataset_shuffled_indices = list(range(dataset_size)) # Shuffle indices for each epoch
current_dataset_filename = dataset_filename # Define current dataset filename
dataset_index = 0 # Initialize dataset_index - not used anymore, but keep for now

batch_input_ids_list = [] # Initialize batch_input_ids_list as a global variable
batch_attention_mask_list = [] # Initialize batch_attention_mask_list as a global variable
cache = None # Initialize cache here
outputs = None # Initialize outputs here
def closure(): # Define closure here, outside the if block
  global batch_input_ids_list # Declare batch_input_ids_list as global
  global batch_attention_mask_list # Declare batch_attention_mask_list as global
  global cache # Declare cache as global
  start_time = time.time()
  i = 0
  optimizer.zero_grad()  #TODO: this belongs in the optimizer..
  for input_ids, attention_mask in zip(batch_input_ids_list, batch_attention_mask_list):
#TODO: on the last iteration, reduce the cache to grad_vector size before grad vector to prevent the gradient from also loading the full chunk size of tokens from the non-differentiable cacheUPDATE: does this matter?
    chunk_size = 200 #200000
    cache=None
    grad_vector_size = 200 #200
    num_tokens = input_ids.size(1)
    cache_position = None
    if chunk_size > 0 :
      for i in range(0, num_tokens - grad_vector_size, chunk_size):
        end_idx = min(i + chunk_size, num_tokens - grad_vector_size)
        cur_input_ids = input_ids[:, i:end_idx]
        cur_attention_mask = attention_mask[:, i:end_idx]
#        print(f"Cache position: {i}")
        if cache is not None:
          with torch.no_grad(): # Keep no_grad context for forward passes in the loop
            outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, cache_params = cache, use_cache = True, cache_position=torch.tensor([i]))
        else:
          with torch.no_grad(): # Keep no_grad context for forward passes in the loop
            outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask, labels = cur_input_ids, use_cache=True)
        cache = outputs.cache_params
#        if not torch.isnan(outputs.loss): # Check for NaN before accumulating
  #        cache_position = cache_position[-1:] + end_idx - i # add one more position for the next token
#      gc.collect()
#      torch.cuda.empty_cache()

#      print(f"Cache position: {num_tokens - grad_vector_size}")
      outputs = model(input_ids[:, -grad_vector_size:], attention_mask=attention_mask[:, -grad_vector_size:],labels = input_ids[:, -grad_vector_size:], cache_params = cache, cache_position=torch.tensor([num_tokens - grad_vector_size]))

    outputs.loss.backward() # Backpropagate gradients

    print(f"{outputs.loss.item():.16f}")
    return outputs.loss

#TODO: save model, indices and fbfgs to the same directory. Consolidate the datasets indices with the model data.
while True: # Main training loop
    cache = None  # Reset cache at the start of each iteration NOTE: globally declared for closure
    dataset_shuffled_indices = list(range(dataset_size)) # Reshuffle indices at the start of each epoch
    random.shuffle(dataset_shuffled_indices) # Reshuffle

    if not dataset_shuffled_indices: # Reshuffle if indices are empty (all seen)
        dataset_shuffled_indices = list(range(dataset_size)) # Recreate full list of indices
        random.shuffle(dataset_shuffled_indices) # Reshuffle
        seen_indices = [] # Reset seen indices when reshuffling all

    if not dataset_shuffled_indices: # Double check in case dataset is empty
        print("Dataset is empty, stopping training for this dataset.")
        break # Exit loop if dataset is empty

    dataset_idx = dataset_shuffled_indices.pop() # Get and remove last index (more efficient than pop(0))
    while dataset_idx in seen_indices and dataset_shuffled_indices: # Skip if index already seen and there are more indices
        dataset_idx = dataset_shuffled_indices.pop() # Get next index
    if dataset_idx in seen_indices: # If still seen (dataset exhausted or all seen), reshuffle and continue
        print("All indices seen, reshuffling and continuing.")
        dataset_shuffled_indices = list(range(dataset_size))
        random.shuffle(dataset_shuffled_indices)
        seen_indices = []
        continue # Go to the next iteration with reshuffled indices

    batch_input_ids_list = [] # Initialize lists for the new batch
    batch_attention_mask_list = []
    batch_count = 0 # Counter for the number of data points in the current batch
    while batch_count < batch_size: # Continue until batch_size is reached
        if not dataset_shuffled_indices: # Check if indices are exhausted during batch collection
            print(f"Dataset indices exhausted before filling batch. Current batch size: {batch_count}")
            break # Break inner loop if no more indices

        dataset_idx = dataset_shuffled_indices.pop()
        while dataset_idx in seen_indices and dataset_shuffled_indices:
            dataset_idx = dataset_shuffled_indices.pop()
        if dataset_idx in seen_indices: # If still seen after trying to find unseen, break batch collection
            print("All indices seen, ending batch collection early.")
            break # Break inner loop if no more unseen indices

        seen_indices.append(dataset_idx) # Mark index as seen
        #print(f"Processing dataset index: original index: {dataset_idx}, unseen indices remaining: {len(dataset_shuffled_indices)}")
        batch_train = dataset[dataset_idx]['code']
        #print(str(batch_train))
        tokens = tokenizer(batch_train,truncation=False, max_length=None,padding=False, return_overflowing_tokens=False, return_length=True,return_tensors='pt').to("cuda")
        input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)
        print("got num_tokens: " + str(input_ids.size(1)))

        current_num_tokens = input_ids.size(1)

        # Truncate to a random size between 200 and 2000 tokens if longer
        max_len_global = random.randint(2000, 20000)
        if current_num_tokens > max_len_global:
            start_idx = random.randint(0, current_num_tokens - max_len_global)
            input_ids = input_ids[:, start_idx : start_idx + max_len_global]
            attention_mask = attention_mask[:, start_idx : start_idx + max_len_global]
            current_num_tokens = input_ids.size(1)
            #print(f"Truncated index {dataset_idx} to random {max_len_global} tokens. New length: {current_num_tokens}")

        # Warmup period truncation
        max_warmup_length = 200
        if len(seen_indices) < 0 and current_num_tokens > max_warmup_length:
            start_idx = random.randint(0, current_num_tokens - max_warmup_length)
            input_ids = input_ids[:, start_idx : start_idx + max_warmup_length]
            attention_mask = attention_mask[:, start_idx : start_idx + max_warmup_length]
            current_num_tokens = input_ids.size(1)
            #print(f"Truncated index {dataset_idx} to random {max_warmup_length} tokens during warmup. New length: {current_num_tokens}")

        # Skip if token length is less than 5 after all truncations
        if current_num_tokens < max_warmup_length:
            print(
                #f"Skipping index {dataset_idx} due to token length ({current_num_tokens}) being less than warmup length."
            )
            continue  # Skip to the next iteration of the inner while loop

        batch_input_ids_list.append(input_ids)
        batch_attention_mask_list.append(attention_mask)
        batch_count += 1 # Increment batch_count only when a valid datapoint is added

    #print(f"--- Before generate - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #print(f"--- Before generate - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    prompt = "-- A Haskell Module that opens a file and prints it to stdout:"
    out = tokenizer(prompt, return_tensors="pt").to("cuda") # Ensure input is on the same device as the model
    with torch.no_grad():
      #print("generating..")
      model.eval()
      generated_ids = model.generate(out.input_ids, max_new_tokens=5, attention_mask=out.attention_mask) # Reduced max_length for debugging
      model.train()
      #print("generation complete:")
      #print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))
      generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
      #print(f"Model response: {generated_text}")

    #print("-----------------------step---------------------")
    # Print loss before optimizer step
    loss_before = closure()
    print(f"Loss before step: {loss_before:.16f}")

    # Perform optimizer step
    optimizer.step(closure)

    # Create RGB color step display
    step_text = f" STEP {step_count} "
    color_cycle = ["\033[38;2;255;0;0m", "\033[38;2;0;255;0m", "\033[38;2;0;0;255m"]  # RGB colors
    reset = "\033[0m"
    
    # Build rainbow line
    line = ""
    for i in range(32):
        line += f"{color_cycle[i % 3]}-{reset}"
    line += f"{color_cycle[step_count % 3]}{step_text}{reset}"
    for i in range(32):
        line += f"{color_cycle[(i+step_count) % 3]}-{reset}"
    
    print(f"\n{line}\n")
    
    # Print loss after optimizer step
    loss_after = closure()
    
    # Show delta in gray text
    loss_delta = loss_before - loss_after
    print(f"\033[90mLoss delta gap: {loss_delta:.16f}\033[0m")
    
    # Collect data for plots
    step_data.append(step_count)
    losses_before.append(loss_before.item())
    losses_deltas.append(loss_delta.item())
    
    # Update plots
    # Clear and update both subplots
    ax1.cla()
    ax2.cla()
    
    # Plot loss before on top subplot
    ax1.plot(step_data, losses_before, 'b-')
    ax1.set_title(f"Loss Before vs Steps (Step {step_count})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    
    # Plot loss delta on bottom subplot
    ax2.plot(step_data, losses_deltas, 'r-')
    ax2.set_title(f"Loss Delta vs Steps (Step {step_count})")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss Delta")
    
    # Save combined figure
    plt.savefig("training.png")

    # Log to CSV file
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
        #      unwrapped_model = accelerator.unwrap_model(model)
        current_dataset_filename = dataset_filename  # Define current dataset filename
        dataset_indices[current_dataset_filename] = seen_indices
        if accelerator.is_main_process:  # Ensure save only on main process
            model.save_pretrained(filename, safe_serialization=False)  # Only save Peft adapter
            #print("model saved..")
            torch.save(dataset_indices, indices_filename)
            #print("indices saved..")
            optimizer.save_history(history_filename)
            #print("optimizer checkpoint saving commented out..")
            print(
                f"Model, indices, and FBFGS history saved to {filename}, {indices_filename}, and {history_filename} at step {step_count}, seen indices count for {current_dataset_filename}: {len(seen_indices)}"
            )
#EOF
