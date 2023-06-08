import subprocess

# Define a list of commands to run
commands = [
    "python Continual_Delay/main_l2ml.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym02_w100_12_3_975_10'",
    "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym01_w100_12_3_975_10'",
    "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 150 --ws 100 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym005_w100_12_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './data3/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym02_w150_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym01_w150_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym005_w150_35_3_975_10'",
    # "python AE_Detector/mainae.py --data './data3/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae02_gn001_w100_ws100_thr125'",
    # "python AE_Detector/mainae.py --data './01lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae01_gn001_w100_ws100_thr125'",
    # "python AE_Detector/mainae.py --data './005lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae005_gn001_w100_ws100_thr125'",
    "python Continual_Delay/main_l2ml.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym02_w50_12_3_975_10'",
    "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 100 --ws 50 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym01_w50_12_3_975_10'",
    "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 100 --ws 50 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delaym005_w50_12_3_975_10'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
