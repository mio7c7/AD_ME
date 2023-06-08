import subprocess

# Define a list of commands to run
COMMANDS=[
    "Continual_Delay/main_mmdr.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.975 --outfile 'delay01_w100_35_3_975_10'"
    "Continual_Delay/main_mmdr.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.95 --outfile 'delay01_w100_35_3_95_10'"
    "Continual_Delay/main_mmdr.py --data './01lr/*.npz' --bs 150 --ws 100 --min_requirement 300 --memory_size 500 --cp_range 10 --threshold 3 --quantile 0.925 --outfile 'delay01_w100_35_3_925_10'"
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
