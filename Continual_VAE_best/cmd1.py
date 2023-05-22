import subprocess

# Define a list of commands to run
commands = [
    # "python Continual_VAE_best/main_qsz.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.95 --outfile 'bestqsz_1'",
    # "python Continual_VAE_best/main_qsz.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'bestqsz_2'",
    # "python Continual_VAE_best/main_qsz.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 5 --quantile 0.95 --outfile 'bestqsz_3'",
    # "python Continual_VAE_best/main_qsz.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 6 --quantile 0.95 --outfile 'bestqsz_4'",
    # "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_5'",
    # "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --ws 40 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_6'",
    # "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.85 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_7'",
    # "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --ws 40 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.85 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_8'",
    # "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --ws 40 --memory_size 100 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_9'",
    # "python Continual_VAE_best/main_wpre.py --data './data3/*.npz' --ws 40 --memory_size 200 --dense_dim 2 --dropout 0.5 --cp_range 6 --forgetting_factor 0.75 --out_threshold 2 --threshold 5 --quantile 0.92 --outfile 'bestctnqt_10'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'bestqs_12'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.65 --out_threshold 2 --threshold 5 --quantile 0.95 --outfile 'bestqs_13'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'bestqs_14'",
    # "python Continual_VAE_best/main_qs.py --data './data3/*.npz' --ws 40 --min_requirement 200 --memory_size 400 --dense_dim 2 --dropout 0.5 --cp_range 5 --forgetting_factor 0.5 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'bestqs_15'",
    # "python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 20 --fixed_outlier 2 --threshold 5 --outfile 'wpre_bo20_n20_th5'",
    # "python MStatistics/main_wpre.py --data './data3/*.npz' --bo 40 --N 10 --fixed_outlier 2 --threshold 5 --outfile 'wpre_bo40_n10_th5'",
    # "python MStatistics/main_wpre.py --data './data3/*.npz' --bo 10 --N 40 --fixed_outlier 2 --threshold 5 --outfile 'wpre_bo10_n40_th5'",
    # "python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 20 --fixed_outlier 2 --threshold 4 --outfile 'wpre_bo20_n20_th4'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 10 --threshold 0.6 --outfile 'wpre_bocd_ssawnd5_delay10_threshold06_ip20_bs48'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 10 --threshold 0.7 --outfile 'wpre_bocd_ssawnd5_delay10_threshold07_ip20_bs48'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 10 --threshold 0.5 --outfile 'wpre_bocd_ssawnd5_delay10_threshold05_ip20_bs48'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 20 --threshold 0.6 --outfile 'wpre_bocd_ssawnd5_delay20_threshold06_ip20_bs48'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 20 --threshold 0.7 --outfile 'wpre_bocd_ssawnd5_delay20_threshold07_ip20_bs48'",
    # "python BOCDUNI_noreset.py --data 'data3/*.npz' --ssa_window 5 --bs 48 --delay 20 --threshold 0.5 --outfile 'wpre_bocd_ssawnd5_delay20_threshold05_ip20_bs48'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 15 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_1'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --cp_range 15 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_2'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 60 --ws 75 --min_requirement 200 --memory_size 400 --cp_range 10 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_3'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 85 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 10 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_4'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 15 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_5'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --cp_range 15 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_6'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 60 --ws 75 --min_requirement 200 --memory_size 400 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_7'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 85 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 2 --quantile 0.975 --outfile 'bestqsl2_8'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 60 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 15 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.975 --outfile 'bestqsl2_9'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 85 --ws 75 --min_requirement 200 --memory_size 400 --cp_range 15 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.975 --outfile 'bestqsl2_10'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 60 --ws 75 --min_requirement 200 --memory_size 400 --cp_range 10 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.975 --outfile 'bestqsl2_11'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 85 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 10 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.975 --outfile 'bestqsl2_12'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 60 --ws 75 --min_requirement 200 --memory_size 400 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.975 --outfile 'bestqsl2_13'",
    "python Continual_VAE_best/main_qsl2.py --data './data3/*.npz' --bs 85 --ws 50 --min_requirement 200 --memory_size 400 --cp_range 5 --forgetting_factor 0.55 --out_threshold 2 --threshold 3 --quantile 0.975 --outfile 'bestqsl2_14'",
]

# Loop over the commands and submit each one as a batch job
for i, cmd in enumerate(commands):
    # Submit the batch job and get the job ID
    subprocess.run(cmd, shell=True)

    # Print a message when the command is done
    print(f"Task {i + 1} with command '{cmd}' has finished")
