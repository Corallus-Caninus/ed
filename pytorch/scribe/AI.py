#import causal_conv1d
import os

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

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
filename = "AI_Checkpoint.ai"

#------------------------TRAINING-----------------------

import time
#from trl import SFTTraier, SFTConfig
#from peft import LoraConfig

#tokenizer = AutoTokenizer.from_pretrained("Microsoft/phi2")


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", trust_remote_code=True)
#model = MambaForCausalLM.from_pretraied("state-spaces/mamba-130m-hf").to("cuda")

#El Chapo
mamba_config = MambaConfig(hidden_size=150,num_hidden_layers=64,hidden_act="tanh", layer_norm_epsilon=1e-09, residual_in_fp32=True, use_bias=True)
model = MambaForCausalLM(mamba_config).to("cuda")
#model = torch.nn.DataParallel(model, device_ids = [0,1])

#model = MambaForCausalLM.from_pretrained("El-Chapo").to("cuda")
if os.path.exists(filename):
  unwrapped_model = accelerator.unwrap_model(model)
  unwrapped_model.load_state_dict(torch.load(filename))

dataset = load_dataset(path="datasets", split="train", sample_by="paragraph")
print(dataset)
#TODO: label these
#TODO: this is only useful for cross-validation
dataloader_train = DataLoader(dataset, batch_size=5, shuffle=True)
dataloader_train = [x for x in dataloader_train if len(x['text'][0])!=0]
#TODO: proper tokenized dataset like on huggingface tutorials
data_iter_train = iter(dataloader_train)

model.train()

# Define the optimizer
optimizer = LBFGS(model.parameters(), lr=1., history_size=20, tolerance_change=1e-6, max_iter=25, max_eval=50, line_search_fn="strong_wolfe")

batch_train = None
input_ids = None
attention_mask = None
# Define the closure function required by LBFGS
def closure():
  start_time = time.time()
  loss = 0
  optimizer.zero_grad() #TODO: do we still need to do this?
  # Training loop
  outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
  loss = outputs.loss
  print(str(loss) + "-----------loss print-----------" )

  loss.backward()
  torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.)
  end_time = time.time()
  elapsed_time = end_time - start_time
  return loss

null = None
no = None
#null, no, model, optimizer = accelerator.prepare(
#    null, no, model, optimizer
#null, no, no, optimizer = accelerator.prepare(
#    null, no, no, optimizer
#)
num_epochs = 30
for _ in range(0,num_epochs):
  print("iterating epoch..\n\n")
  
  # Perform optimization step
  try:
    batch_train = next(data_iter_train)
  except StopIteration:
    data_iter_train = iter(dataloader_train)
    print("------------EPOCH COMPLETE----------")


  print("-----------------------step---------------------")
  tokenized_input = tokenizer(batch_train['text'],truncation=True,  padding=True,max_length=20, return_overflowing_tokens=True, return_length=True,return_tensors='pt').to("cuda")
  print(batch_train['text'])
  input_ids, attention_mask = (tokenized_input.input_ids, tokenized_input.attention_mask)

  optimizer.step(closure)

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
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), filename)
#TODO: this doesnt save the optimizer state dictionary. Save it here and serialize it seperately. Consider a class to wrap saving the model and optimizer with the accelerate framework.
  


