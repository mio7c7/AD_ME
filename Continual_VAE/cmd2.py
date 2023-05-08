import subprocess

# Define a list of commands to run
commands = [
    "python Continual_VAE_lifelong/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 2 --quantile 0.95 --outfile 'lifelongc_1'",
    "python Continual_VAE_lifelong/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 3 --quantile 0.95 --outfile 'lifelongc_2'",
    "python Continual_VAE_lifelong/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'lifelongc_3'",
    "python Continual_VAE_lifelong/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 3 --quantile 0.97 --outfile 'lifelongc_4'",
    "python Continual_VAE_lifelong/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 3 --quantile 0.99 --outfile 'lifelongc_5'",
    "python Continual_VAE_lifelong/main_wpre.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 2 --quantile 0.97 --outfile 'lifelongc_6'",
    "python Continual_VAE_lifelong/main_wpre.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 2 --quantile 0.99 --outfile 'lifelongc_7'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
