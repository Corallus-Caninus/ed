#import causal_conv1d

#TODO: get this to expose trainning and inference as a basic API to ED then swap it out for llama.cpp once mamba and mamba cuda/Rocm kernels are written

#from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, MambaModel
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, MambaModel
import torch
#from torch.optim import LBFGS
from lbfgs import LBFGS
#torch.set_num_threads(12)
from datasets import load_dataset
import datasets
#import peft

#tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
#model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")#.to("cuda")

#input_ids = tokenizer("Hey how are ya?", return_tensors = "pt")["input_ids"]##.to("cuda")
#out = model.generate(input_ids, max_new_tokens=400)
#print(tokenizer.batch_decode(out))

#------------------------TRAINING-----------------------
import torch
from torch.utils.data import Dataset, DataLoader
import time

#dataset = load_dataset("./TestData.txt")
#dataset = load_dataset("text", data_files={"trai": "TestData.txt", "test": "TestData.txt"}, sample_by="paragraph")
#dataset = load_dataset("text", data_files={"trai": "TestData.txt", "test": "TestData.txt"})
#prit(dataset)
#trai_dataset = dataset['text']
#dataset = load_dataset("text", data_files="TestData.txt")


from datasets import load_dataset
#from trl import SFTTraier, SFTConfig
#from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments


#dataset = CustomTextDataset(str_data, tokeizer, batch_size=4, shuffle=True)
#TODO: shuffle after each epoch
#TODO: use ctags to orgaize datasets for trainning (specifically C code)
#TODO: use mambafor to create a model from scratch

#tokenizer = AutoTokenizer.from_pretrained("Microsoft/phi2")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
#model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", trust_remote_code=True)
#model = MambaForCausalLM.from_pretraied("state-spaces/mamba-130m-hf").to("cuda")

#El Chapo
mamba_config = MambaConfig(hidden_size=150,num_hidden_layers=32,hidden_act="tanh", layer_norm_epsilon=1e-09, residual_in_fp32=True, use_bias=True)
model = MambaForCausalLM(mamba_config).to("cuda")
#TODO: need to update the training mechanism, also how can we put LBFGS across too? Check that data parallel distributes the model not duplicates the model and distributes the data. May be better to distribute lbfgs History and just assume model can fit.
#model = MambaForCausalLM(mamba_config).to("cuda")
#model = torch.nn.DataParallel(model, device_ids = [0,1])

#model = MambaForCausalLM.from_pretrained("El-Chapo").to("cuda")

#TODO: use tokenizer with dataset to create a tokenized dataset and try to make trainning more idiomatic
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
#TODO: how can we exit with optimality instead of infinity?
optimizer = LBFGS(model.parameters(), lr=1, history_size=5, tolerance_change=1e-6, max_iter=25, max_eval=50, line_search_fn="strong_wolfe")

batch_train = None
input_ids = None
attention_mask = None
# Define the closure function required by LBFGS
def closure():
  start_time = time.time()
  loss = 0
  optimizer.zero_grad()
  #----------------------------EOT--------------------------------
  #----------------------------LBFGS---------------------------
  # Training loop
#  tokenized_input = tokenizer(batch['text'], max_length=50, padding=True, truncation=True, return_tensors='pt')#.to("cuda")
  print("forward propping the model..")
  outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
  print("forward propping the model..")
  loss = outputs.loss
  print("------------------------------------------------" + str(loss) + "-----------loss print-----------" )
  print("that was the loss..")

  loss.backward()
  end_time = time.time()
  elapsed_time = end_time - start_time
#TODO: yield? need to batch, for now this should be alright?
  return loss

num_epochs = 15000000
for _ in range(0,num_epochs):
  print("iterating epoch..\n\n")
  
  # Perform optimization step
  try:
    batch_train = next(data_iter_train)
  except StopIteration:
#TODO prabably shuffle and redo the dataset here
    data_iter_train = iter(dataloader_train)
    print("STOP")


  print("-----------------------step---------------------")
#TODO: extract this out of loop
#  tokenized_input = tokenizer(batch_train['text'], padding=True, max_length=50, truncation=True,return_tensors='pt')#.to("cuda")
  tokenized_input = tokenizer(batch_train['text'],truncation=True,  padding=True,max_length=20, return_overflowing_tokens=True, return_length=True,return_tensors='pt').to("cuda")
  print(batch_train['text'])
  input_ids, attention_mask = (tokenized_input.input_ids, tokenized_input.attention_mask)

  optimizer.step(closure)
  optimizer.state["n_iter"]=2

  # Print model response
  prompt = "The Factor programming language is "
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Model response: {generated_text}")

  model.save_pretrained("El-Chapo")
#  try:
#  except:
#    pass
  
#TODO:
  # Evaluate loss for logging purposes
#  with torch.no_grad():
#      outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
#      loss = outputs.loss
#      print(f"Epoch: {epoch}, Loss: {loss.item()}")
#----------------------------EOF---------------------------


