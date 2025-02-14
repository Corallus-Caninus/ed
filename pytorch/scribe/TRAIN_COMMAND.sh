nix-shell --run 'source ../AIVenv/bin/activate &&  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" CUDA_VISIBLE_DEVICES=1,2 accelerate launch  AI.py'
