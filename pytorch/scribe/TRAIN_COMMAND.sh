nix-shell --run 'source AIVenv/bin/activate && export OMP_NUM_THREADS=12 && export TORCH_NUM_THREADS=12 && PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" accelerate launch --gpu_ids "0,1" AI.py'
