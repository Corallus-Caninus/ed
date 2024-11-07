nix-shell --run 'source ../AIVenv/bin/activate && CUDA_VISIBLE_DEVICES=1,2 accelerate launch --use_deepspeed AI.py'
