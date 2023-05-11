import subprocess

# Define a list of commands to run
commands = [
    "python Continual_VAE_lifelong/main_new.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 400 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 7 --outfile 'lifelongcrstd_1'",
    "python Continual_VAE_lifelong/main_new.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 500 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 7 --outfile 'lifelongcrstd_2'",
    "python Continual_VAE_lifelong/main_new.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 400 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 5 --outfile 'lifelongcrstd_3'",
    "python Continual_VAE_lifelong/main_new.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 500 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 5 --outfile 'lifelongcrstd_4'",
    "python Continual_VAE_lifelong/main_new.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 400 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 9 --outfile 'lifelongcrstd_5'",
    "python Continual_VAE_lifelong/main_new.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 500 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 9 --outfile 'lifelongcrstd_6'",
    # "python Continual_VAE/main_q.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 4 --quantile 0.97 --outfile 'mainq_16'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
