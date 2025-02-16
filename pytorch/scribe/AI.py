import os

import torch

print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

import gc
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, MambaModel, Mamba2ForCausalLM, AutoModel 
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from lbfgs import LBFGS
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
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", trust_remote_code=True)
model_id = "AntonV/mamba2-130m-hf"
model_id = "state-spaces/mamba2-130m"
history_filename = "lbfgs_history.pth"

if os.path.exists(filename): # Load model weights and optimizer history
    print(f"Checkpoint file '{filename}' found. Loading model from checkpoint...")
    model = Mamba2ForCausalLM(MambaConfig()).to("cuda") # Initialize model with default config
    try:
        model.load_state_dict(torch.load(filename, weights_only=True))
        print(f"Model checkpoint loaded successfully from '{filename}'.") # Verification message
    except FileNotFoundError:
        print(f"Model checkpoint file '{filename}' not found, even though it was just checked. This is unexpected.")
        model = Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf").to("cuda") # Fallback to AntonV weights
        print(f"Loading initial weights from '{model_id}' instead.")
    except Exception as e:
        print(f"Error loading model checkpoint from '{filename}': {e}. Falling back to initial weights.")
        model = Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf").to("cuda") # Fallback to AntonV weights
        print(f"Loading initial weights from '{model_id}' instead.")
else: # Load initial model weights from AntonV if no checkpoint exists
    print(f"Checkpoint file '{filename}' not found. Loading initial model weights from '{model_id}'...")
    model = Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf").to("cuda")

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("num parameters: " + str(pytorch_total_params))

optimizer = LBFGS(model.parameters(), lr=1., history_size=4.5, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe",gradient_clop=1e-7, direction_clop=1e-5, c1=1e-6, c2=0.9)

if os.path.exists(filename): # Load optimizer history if checkpoint exists
    optimizer.load_history(history_filename)

datalist = []
if os.path.exists("chunked.ds"):
    dataset = datasets.load_from_disk("chunked.ds")
else:
    dataset = load_dataset(path="datasets", split="train")
    print(dataset)
    i = 0
    batch_train = ""
    def encode(examples):
        global batch_train
        global i
        if i >=300:
          res = batch_train
          batch_train = ""
          i = 0
          return {"text": res}
        else:
          i += 1
          batch_train += examples['text']
          return {"text": None}
    dataset = dataset.map(encode)
    dataset = dataset.filter(lambda item: item['text'] != None )
    dataset.save_to_disk("chunked.ds")
model.train()

batch_train = None
input_ids = None
attention_mask = None

total_loss = 0. # Initialize total_loss OUTSIDE closure to accumulate across steps

def closure():
  total_loss
  start_time = time.time()
  loss = 0
  optimizer.zero_grad()  #TODO: this belongs in the optimizer..
  cache = None
  chunk_size=1000 #1000
  grad_vector_size = 100 #5
  num_tokens = input_ids.size(1)
  num_steps = 0
  avg_loss = 0.
  if num_tokens == chunk_size+1:
    chunk_size += 1
  torch.cuda.empty_cache()
  for i in range(0, num_tokens - grad_vector_size, chunk_size):
    end_idx = min(i + chunk_size, num_tokens - grad_vector_size)
    cur_input_ids = input_ids[:, i:end_idx]
    cur_attention_mask = attention_mask[:, i:end_idx]

    if cache is not None:
      with torch.no_grad(): # Keep no_grad context for forward passes in the loop
        outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids, cache_params = cache, use_cache=True,  cache_position=[i])
    else:
      with torch.no_grad(): # Keep no_grad context for forward passes in the loop
        outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids,  use_cache=True)
    cache = outputs.cache_params
    num_steps += 1
    current_loss = outputs.loss.item()
    avg_loss += current_loss # Accumulate loss values

  outputs = model(input_ids[:, -grad_vector_size:], attention_mask=attention_mask[:, -grad_vector_size:],labels = input_ids[:, -grad_vector_size:], cache_params = cache, cache_position=[i])
  last_chunk_loss = outputs.loss.item()
  avg_loss += last_chunk_loss # Accumulate loss from the last chunk as well
  avg_loss = avg_loss / (num_steps) # Calculate average loss (including last chunk)
  outputs.loss.item = avg_loss # Assign average loss value to outputs.loss.item
  loss = outputs.loss # Perform backward pass on the original outputs.loss tensor
  loss.backward()

  print("-", end="")
  end_time = time.time()
  elapsed_time = end_time - start_time
  del cache
  del outputs
  torch.cuda.empty_cache()
  return loss

num_iters = 1000
step_count = 0
dataset_size = len(dataset) # Get dataset size outside the loop

while True:
  random_index = torch.randint(0, dataset_size, (1,)).item() # Generate a random index
  batch_train = dataset[random_index]['text'] # Access data using random index

  tokens = tokenizer(batch_train,truncation=True, max_length=2100,padding=False, return_overflowing_tokens=False, return_length=True,return_tensors='pt').to("cuda")
  input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)
  print("got num_tokens: " + str(input_ids.size(1)))

  print("-----------------------step---------------------")
  optimizer.step(closure)
  step_count += 1

  if step_count % 1 == 0:
      unwrapped_model = accelerator.unwrap_model(model)
      accelerator.save(unwrapped_model.state_dict(), filename)
      optimizer.save_history(history_filename)
      print(f"Model and LBFGS history saved to {filename} and {history_filename} at step {step_count}")

  torch.cuda.empty_cache()
  prompt = "The Factor programming language is "
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids .to("cuda")
  with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Model response: {generated_text}")

unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), filename)
