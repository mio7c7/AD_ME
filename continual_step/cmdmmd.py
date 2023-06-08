import subprocess

# Define a list of commands to run
# "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 400 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 'delay02_w100_41_3_975_10'",
# "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 400 --memory_size 100 --threshold 3 --quantile 0.99 --outfile 'delay02_w100_41_3_99_10'",
# "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 400 --memory_size 100 --threshold 3 --quantile 0.95 --outfile 'delay02_w100_41_3_95_10'",
# "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 50 --threshold 3 --quantile 0.975 --outfile 'delay02_w100_505_3_975_10'",
commands = [
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay02_w25_51_3_975_101'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.95 --outfile 'delay02_w50_51_3_975_101'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay02_w75_51_3_975_102",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile delay02_w100_51_3_975_102'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay02_w125_51_3_975_101'",

    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay02_w25_51_3_975_102'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.95 --outfile 'delay02_w50_51_3_975_102'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay02_w75_51_3_975_103",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile delay02_w100_51_3_975_103'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay02_w125_51_3_975_102'",

    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay02_w25_51_3_975_103'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.95 --outfile 'delay02_w50_51_3_975_103'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay02_w75_51_3_975_104",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile delay02_w100_51_3_975_104'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay02_w125_51_3_975_103'",

    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay02_w25_51_3_975_104'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.95 --outfile 'delay02_w50_51_3_975_104'",
    "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay02_w125_51_3_975_104'",

    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay02_w25_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay01_w25_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay005_w25_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.95 --outfile 'delay02_w50_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.95 --outfile 'delay01_w50_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz'--bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.95 --outfile 'delay005_w50_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay02_w75_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay01_w75_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay005_w75_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.95 --outfile 1delay02_w100_51_3_95_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.95 --outfile 'delay01_w100_51_3_95_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.95 --outfile 'delay005_w100_51_3_95_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay02_w125_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay01_w125_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay005_w125_51_3_975_10'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")


    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay02_w25_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay01_w25_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 75 --ws 25 --min_requirement 500 --memory_size 400 --threshold 3 --quantile 0.975 --outfile 'delay005_w25_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile 'delay02_w50_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile 'delay01_w50_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz'--bs 100 --ws 50 --min_requirement 500 --memory_size 200 --threshold 3 --quantile 0.975 --outfile 'delay005_w50_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay02_w75_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay01_w75_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 125 --ws 75 --min_requirement 500 --memory_size 133 --threshold 3 --quantile 0.975 --outfile 'delay005_w75_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 1delay02_w100_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 'delay01_w100_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 150 --ws 100 --min_requirement 500 --memory_size 100 --threshold 3 --quantile 0.975 --outfile 'delay005_w100_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './data3/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay02_w125_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './01lr/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay01_w125_51_3_975_10'",
    # "python continual_step/main_mmdr.py --data './005lr/*.npz' --bs 150 --ws 125 --min_requirement 500 --memory_size 80 --threshold 3 --quantile 0.975 --outfile 'delay005_w125_51_3_975_10'",