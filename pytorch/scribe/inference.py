import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaConfig, Mamba2ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

model_id = "mistralai/Mamba-Codestral-7B-v0.1" # Or your desired base model ID
filename = "AI_Checkpoint.ai" # Path to your LoRa adapter checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

lora_config =  LoraConfig( # Define LoraConfig explicitly to match training
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        lora_alpha=8,
        bias="lora_only",
)

if os.path.exists(filename): # Load model weights and LoRa adapter
    print(f"Checkpoint file '{filename}' found. Loading LoRa adapter from checkpoint...")
    config = MambaConfig.from_pretrained(model_id, trust_remote_code=True) # Load base config
    model = Mamba2ForCausalLM.from_pretrained(model_id, config=config,  torch_dtype=torch.float16, ignore_mismatched_sizes=True, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, filename) # Load LoRa weights
    print(f"LoRa adapter loaded successfully from '{filename}'.")
else:
    print(f"Checkpoint file '{filename}' not found. Please ensure the LoRa adapter checkpoint exists.")
    exit()

model = model.to(dtype=torch.float16).to("cuda")
model.eval() # Set model to evaluation mode

prompt = "-- A Haskell Module that opens a file and prints it to stdout:" # Or your desired prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

print(f"--- Before generate - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"--- Before generate - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

print(f"--- After generate - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"--- After generate - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

print("\nGenerated Text:")
print(generated_text)
