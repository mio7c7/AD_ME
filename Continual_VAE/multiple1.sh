#!/bin/bash
#SBATCH --job-name=akari
#SBATCH --output=akari.out
#SBATCH --error=akari.err
#SBATCH --array=1-15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Define the maximum number of jobs to run in parallel
MAX_JOBS=3

# Define the current batch number
BATCH=$((SLURM_ARRAY_TASK_ID / MAX_JOBS + 1))

# Define the index of the job within the current batch
JOB_INDEX=$((SLURM_ARRAY_TASK_ID % MAX_JOBS))

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

# Calculate the start and end index for the current batch
START_INDEX=$((MAX_JOBS * (BATCH - 1)))
END_INDEX=$((START_INDEX + MAX_JOBS - 1))

# If this is the last batch, set the end index to the last command
if [[ $END_INDEX -gt $((${#COMMANDS[@]} - 1)) ]]; then
    END_INDEX=$((${#COMMANDS[@]} - 1))
fi

# Wait until there are less than the maximum number of jobs running
while [[ $(squeue -u $USER | grep myjob | wc -l) -ge $MAX_JOBS ]]; do
    sleep 10
done

# Run the job
${COMMANDS[$START_INDEX+$JOB_INDEX]}

# If this is the last job in the batch, submit the next batch of jobs
if [[ $JOB_INDEX -eq $(($MAX_JOBS - 1)) ]]; then
    sbatch --array=$(($START_INDEX+$MAX_JOBS+1))-$((END_INDEX+1)) $0
fi
