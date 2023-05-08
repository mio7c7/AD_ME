#!/bin/bash
#SBATCH --job-name=akari
#SBATCH --output=akari.out
#SBATCH --error=akari.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=5

# Load required modules
module load mymodule

# Define the number of parallel tasks to run
NUM_TASKS=8

# Define the commands to run in parallel
COMMANDS=(
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae1'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 7 --outfile 'cvae2'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 8 --outfile 'cvae3'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 6 --outfile 'cvae4'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae5'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae6'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae7'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 0.1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae8'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 7 --outfile 'cvae9'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 6 --outfile 'cvae10'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 400 --out_threshold 2 --threshold 6 --outfile 'cvae11'"
    "Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 4 --kl_weight 1 --epoch 400 --out_threshold 2 --threshold 6 --outfile 'cvae12'"
    "OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.9 --guard_zone 0.97 --p 10 --outfile 'oeco1'"
    "OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.97 --p 10 --outfile 'oeco2'"
    "OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 20 --p 10 --outfile 'ff9_sp20_p10'"
)

# Run the parallel tasks using GNU Parallel
srun parallel -j $NUM_TASKS ::: "${COMMANDS[@]}"
