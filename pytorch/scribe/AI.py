#import causal_conv1d
import os
import sys

#TODO: get this to expose trainning and inference as a basic API to ED then swap it out for llama.cpp once mamba and mamba cuda/Rocm kernels are written

#from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, MambaModel
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, MambaModel #AutoModelForCausalLM
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
#from torch.optim import LBFGS
from lbfgs import LBFGS
#torch.set_num_threads(12)
from datasets import load_dataset
import datasets
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
#from mamba_ssm import Mamba2

#fsdp_plugin = FullyShardedDataParallelPlugin(
#    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
#    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
#)
#
#accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
accelerator = Accelerator()
filename = "AI_Checkpoint.ai"

#------------------------TRAINING-----------------------

import time
#from trl import SFTTraier, SFTConfig
#from peft import LoraConfig

#tokenizer = AutoTokenizer.from_pretrained("Microsoft/phi2")


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", trust_remote_code=True)
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to("cuda")
#model = Mamba2.from_pretrained("state-spaces/mamba2-370m").to("cuda")
#TODO: try tensor parallelism since we get an error on FSDP due to dimension and ops (most things only try to support transformers)
#model = accelerator.prepare(model)

#El Chapo
#mamba_config = MambaConfig(hidden_size=200,num_hidden_layers=200,hidden_act="tanh", layer_norm_epsilon=1e-09, residual_in_fp32=True, use_bias=True)
#model = MambaForCausalLM(mamba_config).to("cuda")
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("num parameters: " + str(pytorch_total_params))
#model = torch.nn.DataParallel(model, device_ids = [0,1])

#model = MambaForCausalLM.from_pretrained("El-Chapo").to("cuda")
if os.path.exists(filename):
  unwrapped_model = accelerator.unwrap_model(model)
  unwrapped_model.load_state_dict(torch.load(filename))

#dataset = load_dataset(path="datasets", split="train", sample_by="paragraph")
dataset = load_dataset(path="datasets", split="train")
print(dataset)
dataset = dataset.filter(lambda item: item['text'][0]!=0)
#TODO: use dataset functions to avoid loading the dataset into memory 
#TODO: label these
#TODO: this is only useful for cross-validation
dataloader_train = DataLoader(dataset, batch_size=1, shuffle=False)
#dataloader_train = [x for x in dataloader_train if len(x['text'][0])!=0]
#TODO: clump together n number of entries to ensure length is sufficiently long considering the worse case of minimum_line_length*n
from itertools import islice

#dataloader_train = (" ".join(list(islice(iterator, 5))) for iterator in [dataloader_train] if list(islice(iterator, 5)))
#TODO: proper tokenized dataset like on huggingface tutorials

model.train()

# Define the optimizer
optimizer = LBFGS(model.parameters(), lr=1., history_size=24, tolerance_change=1e-9, max_iter=10, max_eval=100, line_search_fn="strong_wolfe")
null = None
no = None
#null, no, model, optimizer = accelerator.prepare(
#    null, no, model, optimizer
optimizer,  dataloader_train = accelerator.prepare(optimizer,  dataloader_train)
data_iter_train = iter(dataloader_train)

batch_train = None
input_ids = None
attention_mask = None
# Define the closure function required by LBFGS
def closure():
  start_time = time.time()
  loss = 0
  optimizer.zero_grad() #TODO: do we still need to do this?
  # Training loop
  outputs = model(input_ids, attention_mask=attention_mask ,labels=input_ids)
  loss = outputs.loss
#  accelerator.backward(loss) #TODO: accelerate.backward() instead, we should then be able to remove the .to("cuda") lines as well
  loss.backward()

	#TODO: extract gradient conditioning to L-BFGS. This seems manditory for the optimizer to not mess up
  #NOTE: overflow mechanism similar to accumulating half precision to full precision but we simulate less than half precision in f32
#  torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=8000000) 
  #NOTE: we normalize to 10 here but essentially this should set the max variance in an update and is sufficient that we dont need weight decay (is there any benefit to weight decay over this? weight decay seems to damage prior data in the network over time)
#NOTE: 10. or 5. is set based on derivative of the activation function for expressivity, in our case tanh keep in mind this is not the max weight but the max movement of a weight.
  torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.)  
#  if accelerator.sync_gradients:
#        accelerator.clip_grad_norm_(model.parameters(), 1e20)

  end_time = time.time()
  elapsed_time = end_time - start_time
#  max_float = torch.finfo(loss.dtype).max
#  loss = torch.where(loss > 8000000., torch.tensor(8000000., dtype=loss.dtype), loss)

  # Example scalar tensor with NaN value
  #scalar_tensor = torch.tensor(float('nan'))
  
  # Check if the scalar tensor is NaN and replace with max float if true
#  if loss.isnan():
#    print("nan clip")
#    loss.fill_(torch.finfo(loss.dtype).max)
  
#  print(str(loss) + "-----------loss print-----------" )
  #  if loss == torch.tensor(float("nan")):
  #    print("NaN clip")
  #    loss = sys.float_info.max
  torch.cuda.empty_cache()
  return loss

num_iters = 1000
#for _ in range(0,num_iters):
while True:
  print("iterating epoch..\n\n")
  
  # Perform optimization step
  try:
#    batch_train = next(data_iter_train)['text']  + next(data_iter_train)['text']  + next(data_iter_train)['text']  + next(data_iter_train)['text']  + next(data_iter_train)['text']
#TODO: fix this....
#    batch_train = next(data_iter_train)['text']  + next(data_iter_train)['text']  + next(data_iter_train)['text']  + next(data_iter_train)['text']  + next(data_iter_train)['text'] + next(data_iter_train)['text'] + next(data_iter_train)['text'] + next(data_iter_train)['text'] + next(data_iter_train)['text']
    batch_train = next(data_iter_train)['text']
    for _ in range(50-1):
      batch_train += next(data_iter_train)['text']
    #TODO: need to concatenate the arrays here. Keep them in batch size of 5 but concatenate entrywise.
    from itertools import islice
    batch_train = [" ".join(batch_train[i:i+50]) for i in range(0, len(batch_train), 50)]
#    batch_train = str((" ".join(list(islice(iterator, 5))) for iterator in [batch_train] if list(islice(iterator, 5))))
  except StopIteration:
#    break
#    data_iter_train = iter(dataloader_train)
    print("------------EPOCH COMPLETE----------")


  tokenized_input = tokenizer(batch_train,truncation=True,  padding=False, return_overflowing_tokens=True, return_length=True,return_tensors='pt').to("cuda")
#  tokenized_input = tokenizer(batch_train, padding=True,truncation=True,return_tensors="pt")
  print(batch_train)
  input_ids, attention_mask = (tokenized_input.input_ids, tokenized_input.attention_mask)
  print("input_ids shape:", input_ids.shape)
  print("attention_mask shape:", attention_mask.shape)

  print("-----------------------step---------------------")
  optimizer.step(closure)
  torch.cuda.empty_cache()

  # Print model response
  prompt = "The Factor programming language is "
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids .to("cuda")
  with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Model response: {generated_text}")
#
#TODO: need to save out the Hessian approx too or we nand (also lower init LR)
#  model.save_pretrained("El-Chapo") #TODO: implement this with the accelerate framework.
#TODO: implement proper serialization for accelerate (does this include the optimizer params? we need those too)
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), filename)
#TODO: this doesnt save the optimizer state dictionary. Save it here and serialize it seperately. Consider a class to wrap saving the model and optimizer with the accelerate framework.
  


