import subprocess

# Define a list of commands to run
commands = [
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w25_51_3_975_101'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w50_51_3_975_101'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w75_51_3_975_101'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w100_51_3_975_101'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w125_51_3_975_101'",

    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w25_51_3_975_102'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w50_51_3_975_102'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w75_51_3_975_102'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w100_51_3_975_102'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w125_51_3_975_102'",

    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w25_51_3_975_103'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w50_51_3_975_103'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w75_51_3_975_103'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w100_51_3_975_103'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w125_51_3_975_103'",

    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w25_51_3_975_104'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w50_51_3_975_104'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w75_51_3_975_104'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w100_51_3_975_104'",
    "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w125_51_3_975_104'",



    # "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w25_51_3_975_10'",
    # "python continual_step/main_qs.py --data './01lr/*.npz' --bs 150 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delayvae01_w25_51_3_975_10'",
    # "python continual_step/main_qs.py --data './005lr/*.npz' --bs 150 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delayvae005_w25_51_3_975_10'",
    # "python continual_step/main_qs.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile '1delayvae02_w50_51_3_975_10'",
    # "python continual_step/main_qs.py --data './01lr/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile '1delayvae01_w50_51_3_975_10'",
    # "python continual_step/main_qs.py --data './005lr/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile '1delayvae005_w50_51_3_975_10'",
    # "python continual_step/main_qs.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile '1delayvae02_w75_51_3_975_10'",
    # "python continual_step/main_qs.py --data './01lr/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile '1delayvae01_w75_51_3_975_10'",
    # "python continual_step/main_qs.py --data './005lr/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile '1delayvae005_w75_51_3_975_10'",
    # "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile '1delayvae02_w100_51_3_975_10'",
    # "python continual_step/main_qs.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile '1delayvae01_w100_51_3_975_10'",
    # "python continual_step/main_qs.py --data './005lr/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile '1delayvae005_w100_51_3_975_10'",
    # "python continual_step/main_qs.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delayvae02_w125_51_3_975_10'",
    # "python continual_step/main_qs.py --data './01lr/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delayvae01_w125_51_3_975_10'",
    # "python continual_step/main_qs.py --data './005lr/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delayvae005_w125_51_3_975_10'",

]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
