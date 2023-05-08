#!/bin/bash
#SBATCH --job-name=ami
#SBATCH --output=ami.out
#SBATCH --error=ami.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6

# Define the number of parallel tasks to run
NUM_TASKS=6

# Define the commands to run in parallel
COMMANDS=(
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae5'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae6'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae7'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 0.1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae8'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 7 --outfile 'cvae9'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 6 --outfile 'cvae10'"
)

# Run the parallel tasks using GNU Parallel
srun parallel -j $NUM_TASKS ::: "${COMMANDS[@]}"
