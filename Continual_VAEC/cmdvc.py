import subprocess

# Define a list of commands to run
commands = [
    "python Continual_VAEC/main_qs.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 25 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'VC_w50_2975_25'",
    "python Continual_VAEC/main_qs.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 25 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'VC_w50_2975_25'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
