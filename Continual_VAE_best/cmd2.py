import subprocess

# Define a list of commands to run
commands = [
    "python Continual_VAE_best/main.py --data './data3/*.npz' --bs 50 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.9 --outfile 'bestsqt_1'",
    "python Continual_VAE_best/main.py --data './data3/*.npz' --bs 50 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestsqt_2'",
    "python Continual_VAE_best/main.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 6 --quantile 0.9 --outfile 'bestsqt_3'",
    "python Continual_VAE_best/main.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 4 --quantile 0.9 --outfile 'bestsqt_4'",
    "python Continual_VAE_best/main.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 100 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.9 --outfile 'bestsqt_5'",
    "python Continual_VAE_best/main.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 100 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestsqt_6'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
