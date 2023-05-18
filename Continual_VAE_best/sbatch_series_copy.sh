#!/bin/bash
for G in 1
do
#    sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.9 --guard_zone 0.95 --p 10 --outfile 'ff9_sp30_p10_ot2_nb09_gz095'
#    sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.9 --p 10 --outfile 'ff9_sp30_p10_ot2_nb085_gz09'
#    sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.92 --guard_zone 0.96 --p 10 --outfile 'ff9_sp30_p10_ot2_nb092_gz096'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr8_latent1_batchsize32_epoch50_ot2'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8.5 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr85_latent1_batchsize32_epoch50_ot2'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 7.5 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr75_latent1_batchsize32_epoch50_ot2'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 6 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws20_thr6_latent1_batchsize32_epoch100_ot2'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 7 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws20_thr65_latent1_batchsize32_epoch100_ot2'
#    sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws20_thr55_latent1_batchsize32_epoch100_ot2'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 400 --out_threshold 2 --threshold 6 --outfile 'cvae11'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 4 --kl_weight 1 --epoch 400 --out_threshold 2 --threshold 6 --outfile 'cvae12'
#     sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.9 --guard_zone 0.97 --p 10 --outfile 'oeco1'
#     sbatch sbatch2.sh OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.97 --p 10 --outfile 'oeco2'
#     sbatch sbatch2.sh OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 20 --p 10 --outfile 'ff9_sp20_p10'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 6 --outfile 'cvae10'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 7 --outfile 'cvae9'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 0.1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae8'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae7'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae6'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae5'
#     sbatch sbatch2.sh Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 6 --outfile 'cvae4'
     sbatch sbatch2.sh Continual_VAE/main8.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 3 --quantile 0.95 --outfile 'main8_1'
     sbatch sbatch2.sh Continual_VAE/main8.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 4 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 4 --quantile 0.95 --outfile 'main8_2'
     sbatch sbatch2.sh Continual_VAE/main8.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 4 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 3 --quantile 0.96 --outfile 'main8_3'
#      sbatch sbatch2.sh cmdbs.py
#      sbatch sbatch2.sh cmd2.py
#      sbatch sbatch2.sh cmd3.py
done
