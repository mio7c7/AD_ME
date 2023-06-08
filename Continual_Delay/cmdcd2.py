import subprocess

# Define a list of commands to run
commands = [
    # "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayml01_w100_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.95 --outfile 'delayml01_w100_35_3_95_10'",
    # "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 100 --ws 50 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delay01_w50_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 100 --ws 50 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.95 --outfile 'delay01_w50_35_3_95_10'",
    # "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delay01_w150_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './01lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.95 --outfile 'delay01_w150_35_3_95_10'",
    # "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 150 --ws 100 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delay005_w100_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 150 --ws 100 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.95 --outfile 'delay005_w100_35_3_95_10'",
    # "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 100 --ws 50 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delay005_w50_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 100 --ws 50 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.95 --outfile 'delay005_w50_35_3_95_10'",
    # "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delay005_w150_35_3_975_10'",
    # "python Continual_Delay/main_l2ml.py --data './005lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.95 --outfile 'delay005_w150_35_3_95_10'",
    "python Continual_Delay/main_qs.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w100_12_3_975_10'",
    "python Continual_Delay/main_qs.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae01_w100_12_3_975_10'",
    "python Continual_Delay/main_qs.py --data './005lr/*.npz' --bs 150 --ws 100 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae005_w100_12_3_975_10'",
    # "python Continual_Delay/main_qs.py --data './data3/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w150_35_3_975_10'",
    # "python Continual_Delay/main_qs.py --data './01lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae01_w150_35_3_975_10'",
    # "python Continual_Delay/main_qs.py --data './005lr/*.npz' --bs 200 --ws 150 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae005_w150_35_3_975_10'",
    "python Continual_Delay/main_qs.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w50_12_3_975_10'",
    "python Continual_Delay/main_qs.py --data './01lr/*.npz' --bs 100 --ws 50 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae01_w50_12_3_975_10'",
    "python Continual_Delay/main_qs.py --data './005lr/*.npz' --bs 100 --ws 50 --min_requirement 100 --memory_size 200 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delayvae005_w50_12_3_975_10'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
