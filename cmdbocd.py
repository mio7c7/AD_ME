import subprocess

# Define a list of commands to run
commands = [
    "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 100 --threshold 0.4 --outfile 'bocd02_ssawnd5_delay100_threshold04_ip50_bs48'",
    "python BOCDUNI_noreset.py --data '01lr/*.npz' --ssa_window 5 --bs 48 --delay 100 --threshold 0.4 --outfile 'bocd01_ssawnd5_delay100_threshold04_ip50_bs48'",
    "python BOCDUNI_noreset.py --data '005lr/*.npz' --ssa_window 5 --bs 48 --delay 100 --threshold 0.4 --outfile 'bocd005_ssawnd5_delay100_threshold04_ip50_bs48'",
    "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 100 --threshold 0.4 --outfile 'wpre_bocd_ssawnd5_delay100_threshold06_ip50_bs48'",
    "python AE_Detector/mainae.py --data './data3/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae02_gn001_w100_ws100_thr125'",
    "python AE_Detector/mainae.py --data './01lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae01_gn001_w100_ws100_thr125'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 20 --threshold 0.7 --outfile 'wpre_bocd_ssawnd5_delay20_threshold07_ip20_bs48'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 20 --threshold 0.5 --outfile 'wpre_bocd_ssawnd5_delay20_threshold05_ip20_bs48'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")

