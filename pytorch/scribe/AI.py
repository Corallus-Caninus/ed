import os
import sys

import loralib as lora
import torch


print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

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
#from mamba_ssm import Mamba2
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
#TODO:  save/load the model and lbfgs history every n number of data iterations.
#TODO: add LoRa and/or QLoRa so all the kids will try this and not gripe about the scaling
#TODO: save and load the indices to a seperate file so we can verify.
#TODO: project Basilisk: parallelize the model layer-wise with the gradients. Also parallelize the flat-grads and gtd etc in L-BFGS-N. Simplest parallelization, assuming we are using commodity last-gen accelerators for edge learning, this will allow the most performant scale-out of models (e.g.: 3 k80's or 3 MI25's)

import time
#model_id = "state-spaces/mamba2-130m"
model_id = "mistralai/Mamba-Codestral-7B-v0.1"
dataset_filename = "haskell_code_dataset.ds"
model_id = "hanzla/Falcon3-Mamba-R1-v0"
model_id = "state-spaces/mamba2-370m"
model_id = "AntonV/mamba2-1.3b-hf" # No longer needed, using state-spaces/mamba2-130m consistently
model_id = "AntonV/mamba2-2.7b-hf" # No longer needed, using state-spaces/mamba2-130m consistently
#model_id = "state-spaces/mamba-2.8b"
#model_id = "AntonV/mamba2-130m-hf" # No longer needed, using state-spaces/mamba2-130m consistently
#model_id = "tiiuae/falcon-mamba-7b"
#model_id = "state-spaces/mamba-1.4b-hf"
history_filename = "fbfgs_history.pth"
indices_filename = "dataset_indices.pth"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained(model_id, from_slow=True, legacy=False)
#tokenizer = AutoTokenizer.from_pretrained(model_id,  legacy=False)
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

if os.path.exists(filename): # Load model weights and optimizer history
    print(f"Checkpoint file '{filename}' found. Loading LoRa adapter from checkpoint...")
    config = MambaConfig.from_pretrained(model_id, trust_remote_code=True) # Load config from pretrained
    lora_config =  LoraConfig(
            r=16,
            target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
            task_type="CAUSAL_LM",
#            lora_alpha=8,
            use_dora=True,
            bias="lora_only",
    )
    #model = AutoModelForCausalLM(config).to("cuda") # Initialize model with config # REMOVE - incorrect instantiation
#    peft_config = PeftConfig.from_pretrained("AI_Checkpoint.ai")
    lora_config = LoraConfig.from_pretrained("AI_Checkpoint.ai")
#    model = Mamba2ForCausalLM.from_pretrained(model_id, config=config,  torch_dtype=torch.float16, ignore_mismatched_sizes=True, device_map="auto")
    model = Mamba2ForCausalLM.from_pretrained(model_id, config=config,  torch_dtype=torch.float16, ignore_mismatched_sizes=True, device_map="auto", trust_remote_code=True)
#    model = PeftModel.from_pretrained(model, filename) # Load Lora weights
#    model = LoraModel(model, lora_config, "default") # Load Lora weights
#    model.load_state_dict(torch.load("AI_Checkpoint.ai/adapter_model.safetensors"), strict=False)
    dataset_indices = {}
    if os.path.exists(indices_filename):
        dataset_indices = torch.load(indices_filename) # Load dataset_indices, default to empty dict

    current_dataset_filename = dataset_filename # Define current dataset filename
    if dataset_indices:
        print("Warning: Checkpoint contains dataset indices, ensure you are using the correct dataset or intend to resume.")
    print("After loading - dataset_indices:", dataset_indices)
    seen_indices = dataset_indices.get(current_dataset_filename, []) # Load seen_indices, default to empty list
    print("After loading - seen_indices:", seen_indices)
    #current_index = dataset_indices.get(current_dataset_filename, 0) # No longer needed
    print(f"Model checkpoint loaded successfully from '{filename}'. Resuming {current_dataset_filename} with {len(seen_indices)} indices seen.")

else:
    print(f"Checkpoint file '{filename}' not found. Loading base model weights from '{model_id}' and initializing LoRa adapter...")
    config = Mamba2Config.from_pretrained(model_id, trust_remote_code=True)
#    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, ignore_mismatched_sizes=True, device_map='balanced', torch_dtype=torch.float16)
    print("--- Model Named Parameters (freshly loaded base model) ---")
    for name, param in model.named_parameters(): # Non-recursive for brevity initially
        print(f"Parameter Name: {name}, Parameter Shape: {param.shape}")
    print("--- End Model Inspection (freshly loaded base model) ---")
#    model = Mamba2ForCausalLM.from_pretrained(model_id, config=config,  torch_dtype=torch.float16, ignore_mismatched_sizes=True, device_map="auto")
#    model = Mamba2ForCausalLM.from_pretrained(config, device_map="auto")
#    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    dataset_indices = {}
    current_dataset_filename = dataset_filename # Define current dataset filename
    seen_indices = [] # Initialize seen_indices for new run
    #current_index = 0 # Initialize current_index to 0 for new runs # No longer needed
#Initialize and apply LoRa config:
lora_config =  LoraConfig(
        r=16,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        use_dora=True,
#        lora_alpha=8,
        bias="lora_only",
#            init_weights = "bat",
#            torch_dtype=torch.float16 ,
#            bias="none",
#            use_rslora=True,
)
#    lora_params = (
##        param for name, param in model.named_parameters()
#        param for name, param in model.named_parameters()
#        if  param.requires_grad
##        if "bone_" in name and param.requires_grad
#    )
#    model = LoraModel(model, lora_config, "default")
model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
model = model.to(dtype=torch.float16)
#Get the params ready for passing as flat_grad to fbfgs
lora_params = (
#        param for name, param in model.named_parameters()
    param for name, param in model.named_parameters()
    if  param.requires_grad
#        if "lora_" in name and param.requires_grad
)

 
batch_size = 1 # Define batch size here
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("num parameters: " + str(pytorch_total_params))

#NOTE: mathematically optimized wolfe condition for exponential decay
optimizer = FBFGS(lora_params, lr=1., history_size=9, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe", norm=1., clop=1e-9, c1=1e-4, c2=(1-0.63212),direction_device="cpu", bracket_shift = 1/3, bracket_shove = 1/3)
#optimizer = FBFGS(lora_params, lr=1., history_size=9, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe", norm=1., clop=1e-9, c1=0.5, c2=(0.9),direction_device="cpu", bracket_shift = 1/3, bracket_shove = 1/3)

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
model.gradient_checkpointing_enable()

batch_train = None

step_count = 0
#dataset_size = len(dataset) # Get dataset size outside the loop
import random

dataset_size = len(dataset)
dataset_shuffled_indices = list(range(dataset_size)) # Shuffle indices for each epoch
#random.shuffle(dataset_shuffled_indices) # Shuffle indices - moved to inside the loop
current_dataset_filename = dataset_filename # Define current dataset filename
dataset_index = 0 # Initialize dataset_index - not used anymore, but keep for now

cache = None # Initialize cache here
batch_input_ids_list = [] # Initialize batch_input_ids_list as a global variable
batch_attention_mask_list = [] # Initialize batch_attention_mask_list as a global variable
def closure(): # Define closure here, outside the if block
  global batch_input_ids_list # Declare batch_input_ids_list as global
  global batch_attention_mask_list # Declare batch_attention_mask_list as global
#  global cache
  total_loss= 0
  start_time = time.time()
  loss = 0
  i = 0
  torch.cuda.empty_cache()
  optimizer.zero_grad()  #TODO: this belongs in the optimizer..
  for input_ids, attention_mask in zip(batch_input_ids_list, batch_attention_mask_list):
    torch.cuda.empty_cache()
#TODO: on the last iteration, reduce the cache to grad_vector size before grad vector to prevent the gradient from also loading the full chunk size of tokens from the non-differentiable cache
    chunk_size=500 #1000
    cache=None
#NOTE: with peft we may be able to scale this arbitrarily as long as we arent adapting the context also embedding layers
    grad_vector_size = 100 #5
    grad_chunk_size = 500
    num_tokens = input_ids.size(1)
    num_steps = 0
    avg_loss = 0.
    cache_position = None
    if num_tokens == chunk_size+1:
      chunk_size += 1
    if chunk_size > 0:
      for i in range(0, num_tokens - grad_vector_size, chunk_size):
        end_idx = min(i + chunk_size, num_tokens - grad_vector_size)
        cur_input_ids = input_ids[:, i:end_idx]
        cur_attention_mask = attention_mask[:, i:end_idx]
        cur_input_ids = cur_input_ids.to("cuda") # Ensure input_ids are on CUDA
        cur_attention_mask = cur_attention_mask.to("cuda") # Ensure attention_mask are on CUDA
#        cache_position = torch.tensor([i])
  
        if cache is not None:
          with torch.no_grad(): # Keep no_grad context for forward passes in the loop
#            cache_position =  torch.tensor(i, dtype=torch.long)
            outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids, cache_params = cache,use_cache = True,cache_position=torch.tensor([i]))
      #            outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids,  use_cache=True)
        else:
    #      with torch.no_grad(): # Keep no_grad context for forward passes in the loop
          with torch.no_grad():
            outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids,  use_cache=True)
    #      outputs.loss.backward()
        cache = outputs.cache_params
        num_steps += 1
        current_loss = outputs.loss
        avg_loss += current_loss # Accumulate loss values
#        cache_position = cache_position[-1:] + end_idx - i # add one more position for the next token
  
      gc.collect()
      torch.cuda.empty_cache()

#      outputs = model(input_ids[:, -grad_vector_size:1+ (-grad_vector_size//2) ], attention_mask=attention_mask[:, -grad_vector_size:1+ (-grad_vector_size//2) ],labels = input_ids[:, -grad_vector_size:1+ (-grad_vector_size//2) ], use_cache=True, cache_params=cache, cache_position=cache_position)
#      loss = outputs.loss # Perform backward pass only on the last grad_vector_size tokens
#      loss.backward()
#      cache = outputs.cache_params # redundant assignment
#      cache_position=torch.tensor([i])
#      outputs = model(input_ids[:, -grad_vector_size//2:], attention_mask=attention_mask[:, -grad_vector_size//2:],labels = input_ids[:, -grad_vector_size//2:], cache_params = cache)
      outputs = model(input_ids[:, -grad_vector_size:], attention_mask=attention_mask[:, -grad_vector_size:],labels = input_ids[:, -grad_vector_size:], cache_params = cache)
      loss = outputs.loss # Perform backward pass only on the last grad_vector_size tokens
      total_loss += loss
      loss.backward()
      cache = outputs.cache_params # redundant assignment
# Process grad_vector_size in chunks of grad_chunk_size
#      start_grad_idx = num_tokens - grad_vector_size
#      for i in range(start_grad_idx, num_tokens, grad_chunk_size):
#          end_grad_idx = min(i + grad_chunk_size, num_tokens)
#          cur_input_ids = input_ids[:, i:end_grad_idx]
#          cur_attention_mask = attention_mask[:, i:end_grad_idx]
#          cur_input_ids = cur_input_ids.to("cuda")
#          cur_attention_mask = cur_attention_mask.to("cuda")
#          outputs = model(input_ids=cur_input_ids, attention_mask=cur_attention_mask, labels=cur_input_ids, cache_params=cache, cache_position=[i])
#          loss = outputs.loss
#          total_loss += loss
#          loss.backward() # Backward pass for each chunk
#          cache = outputs.cache_params # Update cache


    print(str(outputs.loss.item()))
    print(str(avg_loss))
    print(str(outputs.loss))

  print("-", end="") # Indicate step completion
  end_time = time.time() # End time for step duration calculation
  elapsed_time = end_time - start_time
  return total_loss/batch_size


while True:
    cache = None  # Reset cache at the start of each iteration
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
        print(f"Processing dataset index: original index: {dataset_idx}, unseen indices remaining: {len(dataset_shuffled_indices)}")
        batch_train = dataset[dataset_idx]['code']
        print(str(batch_train))
        dataset_index += 1 # Increment dataset_index - not used anymore, but keep for now
        if dataset_index >= dataset_size: # Reset dataset_index - not used anymore, but keep for now
            dataset_index = 0 # Reset dataset_index - not used anymore, but keep for now

        tokens = tokenizer(batch_train,truncation=False, max_length=None,padding=False, return_overflowing_tokens=False, return_length=True,return_tensors='pt').to("cuda")
        input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)
        print("got num_tokens: " + str(input_ids.size(1)))
        if input_ids.size(1) < 1000 :
            print("Skipping datapoint with less than 500 tokens.")
            continue # Skip to the next iteration to find a valid datapoint
        batch_input_ids_list.append(input_ids)
        batch_attention_mask_list.append(attention_mask)
        batch_count += 1 # Increment batch_count only when a valid datapoint is added

    print(f"--- Before generate - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"--- Before generate - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    prompt = "-- A Haskell Module that opens a file and prints it to stdout:"
    out = tokenizer(prompt, return_tensors="pt") .to("cuda")
    with torch.no_grad():
      print("generating..")
      model.eval()
      generated_ids = model.generate(out.input_ids, max_new_tokens=200, attention_mask=out.attention_mask) # Reduced max_length for debugging
      model.train()
      print("generation complete:")
      print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))
      generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
      print(f"Model response: {generated_text}")

    print("-----------------------step---------------------")
    step_count += 1
    optimizer.step(closure)
    torch.cuda.empty_cache()
  
    step_count += 1

    torch.cuda.empty_cache()
    gc.collect()

    if step_count % 1 == 0:
        #      unwrapped_model = accelerator.unwrap_model(model)
        current_dataset_filename = dataset_filename  # Define current dataset filename
        dataset_indices[current_dataset_filename] = seen_indices
        if accelerator.is_main_process:  # Ensure save only on main process
            model.save_pretrained(filename)  # Only save Peft adapter
            print("model saved..")
            torch.save(dataset_indices, indices_filename)
            print("indices saved..")
            optimizer.save_history(history_filename)
            print("optimizer saved..")
            print(
                f"Model, indices, and FBFGS history saved to {filename}, {indices_filename}, and {history_filename} at step {step_count}, seen indices count for {current_dataset_filename}: {len(seen_indices)}"
            )
#TODO: fix this. we get NaN (the history doesnt align). could be lora params not aligning in flat grad or something else. We need a merge and reset without unload operation.
#            model = model.merge_and_unload()  # Merge and unload must be called before re-applying lora
#            model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)  # Re-apply lora
#            model = model.to(dtype=torch.float16)
#        
#
#            # Re-extract lora_params for the *new* LoRa adapter
#            lora_params = (
#                param for name, param in model.named_parameters()
#                if param.requires_grad
#            )
#
#            # Re-initialize optimizer with new LoRa params
#            optimizer = FBFGS(lora_params, lr=1., history_size=9, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe", norm=1., clop=1e-9, c1=1e-8, c2=(1-0.63212),direction_device="cpu", bracket_shift = 1/3, bracket_shove = 1/3)
#            optimizer.load_history(history_filename) # Load history into new optimizer
#
#        
#            # Update the optimizer's parameter groups with the new lora_params
#        #    optimizer.param_groups[0]['params'] = list(lora_params)
#        #    optimizer._params = optimizer.param_groups[0]['params']
#            model.gradient_checkpointing_enable()

#TODO: something broke this, fix it.
  

  

    #unwrapped_model = accelerator.unwrap_model(model) # No longer needed
#    model.save_pretrained(filename) # Only save Peft adapter
