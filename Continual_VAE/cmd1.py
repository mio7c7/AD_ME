import subprocess

# Define a list of commands to run
commands = [
    "python LIFEWATCH/main.py --data './data3/*.npz' --window_size 40 --max_points 500 --min_batch_size 24 --epsilon 0.5 --outfile 'orgws40_mp500_mbs24_eps05'",
    "python LIFEWATCH/main.py --data './data3/*.npz' --window_size 40 --max_points 400 --min_batch_size 24 --epsilon 0.5 --outfile 'orgws40_mp400_mbs24_eps05'",
    "python LIFEWATCH/main.py --data './data3/*.npz' --window_size 40 --max_points 200 --min_batch_size 24 --epsilon 0.5 --outfile 'orgws40_mp200_mbs24_eps05'",
    # "python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.92 --stabilisation_period 40 --p 10 --outfile 'ff92_sp40_p10'",
    # "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 40 --out_threshold 2 --normal_boundary 0.87 --guard_zone 0.97 --p 10 --outfile 'ff9_sp40_p10_ot2_nb87_gz097'",
    # "python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 40 --p 10 --outfile 'ff9_sp40_p10'",
    # "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 40 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.97 --p 10 --outfile 'ff9_sp40_p10_ot2_nb85_gz097'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
