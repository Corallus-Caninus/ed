import os
import sys

import torch


print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

import gc
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer,  AutoModelForCausalLM, AutoConfig

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from fbfgs import FBFGS
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import datasets
from datasets import Dataset
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
#from mamba_ssm import Mamba2

accelerator = Accelerator()
filename = "AI_Checkpoint.ai"
#TODO:  save/load the model and lbfgs history every n number of data iterations.
#TODO: add LoRa and/or QLoRa so all the kids will try this and not gripe about the scaling
#TODO: project Basilisk: parallelize the model layer-wise with the gradients. Also parallelize the flat-grads and gtd etc in L-BFGS-N. Simplest parallelization, assuming we are using commodity last-gen accelerators for edge learning, this will allow the most performant scale-out of models (e.g.: 3 k80's or 3 MI25's)

import time
#model_id = "state-spaces/mamba2-130m"
model_id = "AntonV/mamba2-130m-hf" # No longer needed, using state-spaces/mamba2-130m consistently
dataset_filename = "haskell_code_dataset.ds"
#model_id = "hanzla/Falcon3-Mamba-R1-v0"
history_filename = "fbfgs_history.pth"
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

if os.path.exists(filename): # Load model weights and optimizer history
    print(f"Checkpoint file '{filename}' found. Loading model from checkpoint...")
    config = MambaConfig.from_pretrained(model_id, trust_remote_code=True) # Load config from pretrained
    #model = AutoModelForCausalLM(config).to("cuda") # Initialize model with config # REMOVE - incorrect instantiation
    model = AutoModelForCausalLM.from_pretrained(model_id, ignore_mismatched_sizes=True).to("cuda") # Load initial weights using config, ignoring size mismatches
    checkpoint = torch.load(filename)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False) # Load directly if 'model_state_dict' key is missing
    dataset_indices = checkpoint.get('dataset_indices', {}) # Load dataset_indices, default to empty dict
    current_dataset_filename = dataset_filename # Define current dataset filename
    seen_indices = dataset_indices.get(current_dataset_filename, []) # Load seen_indices, default to empty list
    #current_index = dataset_indices.get(current_dataset_filename, 0) # No longer needed
    print(f"Model checkpoint loaded successfully from '{filename}'. Resuming {current_dataset_filename} with {len(seen_indices)} indices seen.")

else:
    print(f"Checkpoint file '{filename}' not found. Loading initial model weights from '{model_id}'...")
    config = MambaConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, ignore_mismatched_sizes=True).to("cuda")
    dataset_indices = {} # Initialize dataset_indices for new run
    current_dataset_filename = dataset_filename # Define current dataset filename
    seen_indices = [] # Initialize seen_indices for new run
    #current_index = 0 # Initialize current_index to 0 for new runs # No longer needed

batch_size = 10 # Define batch size here
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("num parameters: " + str(pytorch_total_params))

#optimizer = FBFGS(model.parameters(), lr=1., history_size=4.5, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe",gradient_clop=5e-7, direction_clop=1e-5, c1=1e-4, c2=0.9)
#optimizer = FBFGS(model.parameters(), lr=1., history_size=9.5, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe", norm=0.75, clop=5e-11, c1=3e-4, c2=0.9,direction_device="cuda:1", bracket_shift = 1/3, bracket_shove = 1/3)
#NOTE: mathematically optimized wolfe condition for exponential decay
optimizer = FBFGS(model.parameters(), lr=1., history_size=9, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe", norm=1., clop=3e-8, c1=3e-4, c2=(1-0.63212),direction_device="cuda:1", bracket_shift = 1/3, bracket_shove = 1/3)

if os.path.exists(filename): # Load optimizer history if checkpoint exists
    optimizer.load_history(history_filename)

datalist = []
if os.path.exists(dataset_filename):
    if os.path.exists(dataset_filename):
      dataset = datasets.load_from_disk(dataset_filename)
else:
    #dataset = load_dataset("kye/all-torvalds-c-code-1", split="train", name="default")
    dataset = load_dataset("codeparrot/github-code", split="train", name="Haskell-all", streaming=False)
    #dataset = load_dataset("codeparrot/github-code", split="train", name="C-all",streaming=True)
    dataset.save_to_disk(dataset_filename)

model.train()

batch_train = None

num_iters = 1000
step_count = 0
#dataset_size = len(dataset) # Get dataset size outside the loop
import random

dataset_size = len(dataset)
dataset_shuffled_indices = list(range(dataset_size)) # Shuffle indices for each epoch
#random.shuffle(dataset_shuffled_indices) # Shuffle indices - moved to inside the loop
input_ids = None
batch_input_ids_list = []
attention_mask = None
batch_attention_mask_list = []
current_dataset_filename = dataset_filename # Define current dataset filename
current_index = 0 # Initialize current_index to 0
dataset_index = 0 # Initialize dataset_index - not used anymore, but keep for now

cache = None # Initialize cache here
batch_input_ids_list = [] # Initialize batch_input_ids_list as a global variable
batch_attention_mask_list = [] # Initialize batch_attention_mask_list as a global variable
def closure(): # Define closure here, outside the if block
  global  cache # Declare cache as global
  global batch_input_ids_list # Declare batch_input_ids_list as global
  global batch_attention_mask_list # Declare batch_attention_mask_list as global
  total_loss= 0
  start_time = time.time()
  loss = 0
  i = 0
  optimizer.zero_grad()  #TODO: this belongs in the optimizer..
  chunk_size=100 #1000
  grad_vector_size = 10 #5
  num_tokens = input_ids.size(1)
  num_steps = 0
  avg_loss = 0.
  if num_tokens == chunk_size+1:
    chunk_size += 1
  if chunk_size > 0:
    for i in range(0, num_tokens - grad_vector_size, chunk_size):
      end_idx = min(i + chunk_size, num_tokens - grad_vector_size)
      cur_input_ids = input_ids[:, i:end_idx]
      cur_attention_mask = attention_mask[:, i:end_idx]

      if cache is not None:
  #      outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids, cache_params = cache,   cache_position=[i])
  #      outputs.loss.backward()
        if i == 0:
          cache.reset()
        with torch.no_grad(): # Keep no_grad context for forward passes in the loop
          outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids, cache_params = cache, use_cache=True,  cache_position=[i])
      else:
  #      with torch.no_grad(): # Keep no_grad context for forward passes in the loop
        with torch.no_grad(): # Keep no_grad context for forward passes in the loop
          outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids,  use_cache=True)
  #      outputs.loss.backward()
      cache = outputs.cache_params
      num_steps += 1
      current_loss = outputs.loss
      avg_loss += current_loss # Accumulate loss values

    outputs = model(input_ids[:, -grad_vector_size:], attention_mask=attention_mask[:, -grad_vector_size:],labels = input_ids[:, -grad_vector_size:], cache_params = cache, cache_position=[i])
#    last_chunk_loss = outputs.loss
#    avg_loss += last_chunk_loss # Accumulate loss from the last chunk as well
    # If num_steps is 0, avg_loss remains 0, or you can handle it differently if needed.
    # For now, we assume that if no chunks were processed, the loss is just the last chunk loss (or the full loss if no chunking at all).

#  input_ids_grad = input_ids[:, -grad_vector_size:].to("cuda")
#  attention_mask_grad = attention_mask[:, -grad_vector_size:].to("cuda")
#  outputs = model(input_ids_grad, attention_mask=attention_mask_grad, labels=input_ids_grad, cache_params = cache, cache_position=[i]) # Use cache for grad section
  print(str(outputs.loss.item()))
  print(str(avg_loss))
#  if num_steps > 0:
#    avg_loss = avg_loss / num_steps 
#    outputs.loss = avg_loss/(0.1*num_tokens) + outputs.loss
  print(str(outputs.loss))
  loss = outputs.loss # Perform backward pass only on the last grad_vector_size tokens
  loss.backward()

  print("-", end="") # Indicate step completion
  end_time = time.time() # End time for step duration calculation
  elapsed_time = end_time - start_time
  return loss


while True:
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
    for _ in range(batch_size): # Collect batch_size datapoints
        if not dataset_shuffled_indices: # Check if indices are exhausted during batch collection
            break # Break inner loop if no more indices
        dataset_idx = dataset_shuffled_indices.pop()
        while dataset_idx in seen_indices and dataset_shuffled_indices:
            dataset_idx = dataset_shuffled_indices.pop()
        if dataset_idx in seen_indices: # If still seen after trying to find unseen, break batch collection
            print("All indices seen, ending batch collection early.")
            break # Break inner loop if no more unseen indices

        seen_indices.append(dataset_idx) # Mark index as seen
        print(f"Processing dataset index: original index: {dataset_idx}, unseen indices remaining: {len(dataset_shuffled_indices)}")
        batch_train = dataset[dataset_idx]['code']
        print(str(batch_train))
        dataset_index += 1 # Increment dataset_index - not used anymore, but keep for now
        if dataset_index >= dataset_size: # Reset dataset_index - not used anymore, but keep for now
            dataset_index = 0 # Reset dataset_index - not used anymore, but keep for now

        tokens = tokenizer(batch_train,truncation=False, max_length=None,padding=False, return_overflowing_tokens=False, return_length=True,return_tensors='pt').to("cuda")
        input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)
        print("got num_tokens: " + str(input_ids.size(1)))
        if input_ids.size(1) < 500:
            print("Skipping datapoint with less than 500 tokens.")
            continue
        batch_input_ids_list.append(input_ids)
        batch_attention_mask_list.append(attention_mask)

    print("-----------------------step---------------------")
  step_count += 1
  if step_count % 10 == 0:
      unwrapped_model = accelerator.unwrap_model(model)
    optimizer.step(closure)

  step_count += 1
  if step_count % 10 == 0:
      unwrapped_model = accelerator.unwrap_model(model)
      current_dataset_filename = dataset_filename # Define current dataset filename
      dataset_indices[current_dataset_filename] = seen_indices # Update seen_indices list
      checkpoint = {
          'model_state_dict': unwrapped_model.state_dict(),
          'dataset_indices': dataset_indices, # Save dataset_indices dictionary
      }
      torch.save(checkpoint, filename)
      optimizer.save_history(history_filename)
      print(f"Model and FBFGS history saved to {filename} and {history_filename} at step {step_count}, dataset index for {current_dataset_filename}: {current_index}")

  torch.cuda.empty_cache()
  prompt = "--A Haskell file that opens a file and prints it to stdout:"
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids .to("cuda")
  with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Model response: {generated_text}")

unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), filename)
