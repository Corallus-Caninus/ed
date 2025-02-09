import os
import gc
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
from mamba_ssm import Mamba2

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
dataloader_train = DataLoader(dataset, batch_size=1, shuffle=True)

model.train()

optimizer = LBFGS(model.parameters(), lr=1., history_size=65, tolerance_change=16, max_iter=10, max_eval=100, line_search_fn="strong_wolfe",gradient_clop=1e-7, direction_clop=7e-7, c1=1., c2=0.9)
dataloader_train, optimizer = accelerator.prepare( dataloader_train, optimizer)
data_iter_train = iter(dataloader_train)

batch_train = None
input_ids = None
attention_mask = None

def closure():
  start_time = time.time()
  loss = 0
  optimizer.zero_grad()  #TODO: this belongs in the optimizer..
  cache = None
  chunk_size=300 #1000
  grad_vector_size = 100 #5
  num_tokens = input_ids.size(1)
  num_steps = 0
  avg_loss = 0.
 # TODO: Spread the gradient throughout the input vector (every 10 iteration generate gradients with torch.set_grad_enable(True) etc) . However, getting information into the model first is somewhat preferable since we dont clobber the anchor inputs (first N inputs to a recurrent model dont have information)TODO: spread it to prevent vanishing gradient (sparse gradients across the input vector)
#  with torch.no_grad():
  for i in range(0, num_tokens, chunk_size):
    end_idx = min(i + chunk_size, num_tokens )#- grad_vector_size)  # Make sure we don't go beyond the sequence length
    cur_input_ids = input_ids[:, i:end_idx]  # Select tokens i to end_idx
    cur_attention_mask = attention_mask[:, i:end_idx]  # Select the attention mask for the chunk
    
#    outputs = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1],labels = input_ids[:, :-1], use_cache=True)
#TODO: the mamba2 paper says that since its attention like vectorized we dont actually do one token at a time, we need to batch with the attention-like SSM vectorization width
    if cache is not None:
#      outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids, cache_params = cache, use_cache=True, cache_position=[i])
      with torch.no_grad():
        outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids, cache_params = cache, use_cache=True,  cache_position=[i])
      outputs_grad = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids, cache_params = cache,  cache_position=[i])
      outputs_grad.loss.backward()
    else:
      outputs = model(input_ids=cur_input_ids, attention_mask = cur_attention_mask  , labels = cur_input_ids,  use_cache=True)
      outputs.loss.backward()
    cache = outputs.cache_params
    num_steps += 1
    avg_loss += outputs.loss.item()
#  outputs = model(input_ids[:, -grad_vector_size:], attention_mask=attention_mask[:, -grad_vector_size:],labels = input_ids[:, -grad_vector_size:], cache_params = cache, cache_position=[i])
#  loss += outputs.loss.item()
#  loss = loss/num_steps
#  outputs.loss.item = loss
#  outputs.logits = outputs.logits[:, -1:, :]
#  outputs.loss = (outputs.loss + avg_loss) / num_steps
#  print(outputs.loss)
#  loss =  outputs.loss
#  loss.backward()
  loss = avg_loss/num_steps
  print("-", end="")
#  torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1., norm_type=2) #TODO: try just l2 norming them here instead of with clipping
  end_time = time.time()
  elapsed_time = end_time - start_time
  del cache
  del outputs
  torch.cuda.empty_cache()
  return loss

num_iters = 1000
while True:
  batch_train = next(data_iter_train)['text']

  tokens = tokenizer(batch_train,truncation=True, max_length=2001,padding=False, return_overflowing_tokens=False, return_length=True,return_tensors='pt').to("cuda")
  input_ids, attention_mask = (tokens.input_ids, tokens.attention_mask)
  print("got num_tokens: " + str(input_ids.size(1)))

  print("-----------------------step---------------------")
  optimizer.step(closure)

  torch.cuda.empty_cache()
  prompt = "The Factor programming language is "
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids .to("cuda")
  with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Model response: {generated_text}")

unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), filename)
