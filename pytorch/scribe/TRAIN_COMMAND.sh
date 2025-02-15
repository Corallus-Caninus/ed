nix-shell --run 'source ../AIVenv/bin/activate &&  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" accelerate launch --gpu_ids "0,1" AI.py'
