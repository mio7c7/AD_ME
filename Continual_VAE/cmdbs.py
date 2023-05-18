import subprocess

# Define a list of commands to run
commands = [
    "python LIFEWATCH/main.py --data './data3/*.npz' --window_size 40 --max_points 10 --min_batch_size 5 --epsilon 1 --outfile 'orgws40_mp10_mbs5_eps1'",
    "python LIFEWATCH/main.py --data './data3/*.npz' --window_size 20 --max_points 20 --min_batch_size 10 --epsilon 1 --outfile 'orgws20_mp20_mbs10_eps1'",
    "python LIFEWATCH/main.py --data './data3/*.npz' --window_size 10 --max_points 40 --min_batch_size 20 --epsilon 1 --outfile 'orgws10_mp40_mbs20_eps1'",
    # "python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.92 --stabilisation_period 40 --p 10 --outfile 'ff92_sp40_p10'",
    # "python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 40 --p 10 --outfile 'ff9_sp40_p10'",
    "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.65 --stabilisation_period 60 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.95 --p 20 --cs 2 --outfile 'ff65_sp60_p20_c2_nb85_gz095'",
    "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.55 --stabilisation_period 60 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.95 --p 20 --cs 2 --outfile 'ff55_sp60_p20_c2_nb85_gz095'",
    "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.5 --stabilisation_period 60 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.95 --p 20 --cs 2 --outfile 'ff5_sp60_p20_c2_nb85_gz095'",
    "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.65 --stabilisation_period 60 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.95 --p 20 --cs 1.5 --outfile 'ff65_sp60_p20_c15_nb85_gz095'",
    "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.55 --stabilisation_period 60 --out_threshold 2 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 1.5 --outfile 'ff55_sp60_p20_c15_nb9_gz095'",
    "python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.5 --stabilisation_period 60 --out_threshold 2 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 1.5 --outfile 'ff5_sp60_p20_c15_nb9_gz095'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
