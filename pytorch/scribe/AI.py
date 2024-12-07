import os
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, MambaModel, Mamba2ForCausalLM, AutoModel 
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from lbfgs import LBFGS
from datasets import load_dataset
import datasets
from datasets import Dataset
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

accelerator = Accelerator()
filename = "AI_Checkpoint.ai"
#TODO:  save/load the model and lbfgs history every n number of data iterations.
#TODO: add LoRa and/or QLoRa so all the kids will try this and not gripe about the scaling
#TODO: project Basilisk: parallelize the model layer-wise with the gradients. Also parallelize the flat-grads and gtd etc in L-BFGS-N. Simplest parallelization, assuming we are using commodity last-gen accelerators for edge learning, this will allow the most performant scale-out of models (e.g.: 3 k80's or 3 MI25's)

import time
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", trust_remote_code=True)
model_id = "AntonV/mamba2-130m-hf"
model_id = "state-spaces/mamba2-130m"
model = Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf").to("cuda")

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("num parameters: " + str(pytorch_total_params))

if os.path.exists(filename):
  unwrapped_model = accelerator.unwrap_model(model)
  unwrapped_model.load_state_dict(torch.load(filename))

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
        if i >=30:
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
dataloader_train = DataLoader(dataset, batch_size=2, shuffle=True)

model.train()

optimizer = LBFGS(model.parameters(), lr=1., history_size=22, tolerance_change=1e-16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe")
dataloader_train, optimizer = accelerator.prepare( dataloader_train, optimizer)
data_iter_train = iter(dataloader_train)

batch_train = None
input_ids = None
attention_mask = None

def closure():
  start_time = time.time()
  loss = 0
  optimizer.zero_grad()  #TODO: this belongs in the optimizer..
  outputs = model(input_ids, attention_mask=attention_mask ,labels=input_ids)
  loss = outputs.loss
  loss.backward()
  print("-", end="")
#  torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1., norm_type=2) #TODO: try just l2 norming them here instead of with clipping
  end_time = time.time()
  elapsed_time = end_time - start_time
  torch.cuda.empty_cache()
  return loss

num_iters = 1000
while True:
  batch_train = next(data_iter_train)['text']

  tokens = tokenizer(batch_train,truncation=True, max_length=200,padding=True, return_overflowing_tokens=False, return_length=True,return_tensors='pt').to("cuda")
  input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)

  print("-----------------------step---------------------")
  optimizer.step(closure)

  torch.cuda.empty_cache()
  prompt = "The Factor programming language is "
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids .to("cuda")
  with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=20, num_return_sequences=1)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Model response: {generated_text}")

unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), filename)
