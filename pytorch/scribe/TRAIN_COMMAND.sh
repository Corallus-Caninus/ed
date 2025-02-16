nix-shell --run 'source ../AIVenv/bin/activate && export OMP_NUM_THREADS=8 && export TORCH_NUM_THREADS=8 && PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" accelerate launch --gpu_ids "1,2" AI.py'
