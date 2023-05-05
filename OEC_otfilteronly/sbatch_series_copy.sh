#!/bin/bash
for G in 1
do
#    sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.9 --guard_zone 0.95 --p 10 --outfile 'ff9_sp30_p10_ot2_nb09_gz095'
#    sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.9 --p 10 --outfile 'ff9_sp30_p10_ot2_nb085_gz09'
#    sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.92 --guard_zone 0.96 --p 10 --outfile 'ff9_sp30_p10_ot2_nb092_gz096'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr8_latent1_batchsize32_epoch50_ot2'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8.5 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr85_latent1_batchsize32_epoch50_ot2'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 7.5 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr75_latent1_batchsize32_epoch50_ot2'
    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 6 --out_threshold 2.5 --outfile 'wpre_ngn_bs48_ws10_thr6_latent1_batchsize32_epoch50_ot25'
    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 6.5 --out_threshold 2.5 --outfile 'wpre_ngn_bs48_ws10_thr65_latent1_batchsize32_epoch50_ot25'
    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 5.5 --out_threshold 2.5 --outfile 'wpre_ngn_bs48_ws10_thr55_latent1_batchsize32_epoch50_ot25'
done
