import subprocess

# Define a list of commands to run
commands = [
    "python Continual_VAE_best/main_new.py --data './data3/*.npz' --ws 40 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 6 --outfile 'bestctnstd_7'",
    "python Continual_VAE_best/main_new.py --data './data3/*.npz' --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 6 --outfile 'bestctnstd_8'",
    "python Continual_VAE_best/main_new.py --data './data3/*.npz' --ws 40 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.85 --out_threshold 2 --threshold 6 --outfile 'bestctnstd_9'",
    # "python Continual_VAE_best/main_new.py --data './data3/*.npz' --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.85 --out_threshold 2 --threshold 6 --outfile 'bestctnstd_10'",
    "python Continual_VAE_best/main_new.py --data './data3/*.npz' --ws 40 --memory_size 100 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 6 --outfile 'bestctnstd_11'",
    "python Continual_VAE_best/main_new.py --data './data3/*.npz' --ws 40 --memory_size 100 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.85 --out_threshold 2 --threshold 6 --outfile 'bestctnstd_12'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
