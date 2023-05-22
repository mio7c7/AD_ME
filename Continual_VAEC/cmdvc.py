import subprocess

# Define a list of commands to run
commands = [
    "python Continual_VAEC/main_mmdr.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 10 --threshold 1.5 --quantile 0.9 --outfile 'MMD_w50_24_1.5_9_10'",
    "python Continual_VAEC/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 200 --memory_size 400 --cp_range 10 --threshold 1.5 --quantile 0.9 --outfile 'MMD_w100_24_1.5_9_10'",
    "python Continual_VAEC/main_mmdr.py --data './data3/*.npz' --bs 250 --ws 200 --min_requirement 200 --memory_size 400 --cp_range 10 --threshold 1.5 --quantile 0.925 --outfile 'MMD_w200_24_1.5_925_10'",
    # "python Continual_VAEC/main_qsml.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --dense_dim 8 --dropout 0.5 --cp_range 5 --out_threshold 2 --threshold 4 --quantile 0.9 --outfile 'VCSTD_w50_4_5'",
    # "python Continual_VAEC/main_qsml.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --dense_dim 8 --dropout 0.5 --cp_range 5 --out_threshold 2 --threshold 4 --quantile 0.9 --outfile 'VCSTD_w75_4_5'",
    # "python Continual_VAEC/main_qsml.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --dense_dim 8 --dropout 0.5 --cp_range 5 --out_threshold 2 --threshold 1 --quantile 0.925 --outfile 'VC_w50_925_5'",
    # "python Continual_VAEC/main_qsml.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --dense_dim 8 --dropout 0.5 --cp_range 5 --out_threshold 2 --threshold 1 --quantile 0.925 --outfile 'VC_w75_925_5'",
    # "python Continual_VAEC/main_qs.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --dense_dim 16 --dropout 0.5 --cp_range 10 --out_threshold 2 --threshold 1 --quantile 0.85 --outfile 'VC_w50_185_10'",
    # "python Continual_VAEC/main_qs.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --dense_dim 16 --dropout 0.5 --cp_range 10 --out_threshold 2 --threshold 1 --quantile 0.85 --outfile 'VC_w75_185_10'",
    # "python Continual_VAEC/main_qs.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --dense_dim 16 --dropout 0.5 --cp_range 15 --out_threshold 2 --threshold 1 --quantile 0.9 --outfile 'VC_w50_19_15'",
    # "python Continual_VAEC/main_qs.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --dense_dim 16 --dropout 0.5 --cp_range 15 --out_threshold 2 --threshold 1 --quantile 0.9 --outfile 'VC_w75_19_15'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
