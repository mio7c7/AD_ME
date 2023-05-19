import subprocess

# Define a list of commands to run
commands = [
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 100 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.9 --outfile 'bestqs_1'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 100 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_2'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 100 --memory_size 200 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_3'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_4'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.95 --outfile 'bestqs_5'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 6 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_6'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.85 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_7'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_8'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 5 --quantile 0.95 --outfile 'bestqs_9'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_10'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.5 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_11'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_81'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 5 --quantile 0.95 --outfile 'bestqs_91'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_101'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.5 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestqs_111'",
    "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --bs 50 --ws 60 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 10 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.95 --outfile 'bestqs_w50'",
    "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --bs 75 --ws 85 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 10 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.95 --outfile 'bestqs_w75'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'bestqs_w75'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'bestqs_w50'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
