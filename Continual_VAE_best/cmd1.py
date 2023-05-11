import subprocess

# Define a list of commands to run
commands = [
    "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.9 --outfile 'bestctnqt_1'",
    "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_2'",
    "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 6 --quantile 0.9 --outfile 'bestctnqt_3'",
    "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 4 --quantile 0.9 --outfile 'bestctnqt_4'",
    "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 100 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.9 --outfile 'bestctnqt_3'",
    "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --bs 48 --ws 40 --memory_size 100 --dense_dim 2 --dropout 0.5 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_4'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
