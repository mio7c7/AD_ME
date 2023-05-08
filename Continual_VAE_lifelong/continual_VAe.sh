#!/usr/bin/env bash

#python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --p 10 --outfile 'ff9_sp30_p10_ot2'
#python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2.5 --p 10 --outfile 'ff9_sp30_p10_ot25'
#python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 3 --p 10 --outfile 'ff9_sp30_p10_ot3'
#python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.92 --stabilisation_period 30 --out_threshold 2 --p 10 --outfile 'ff92_sp30_p10_ot2'
#python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.92 --stabilisation_period 30 --out_threshold 2.5 --p 10 --outfile 'ff92_sp30_p10_ot25'
#python OEC/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.92 --stabilisation_period 30 --out_threshold 3 --p 10 --outfile 'ff92_sp30_p10_ot3'
#
#python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 7 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr7_latent1_batchsize32_epoch50_ot2'
#python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 7 --out_threshold 2.5 --outfile 'wpre_ngn_bs48_ws10_thr7_latent1_batchsize32_epoch50_ot25'
#python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr8_latent1_batchsize32_epoch50_ot2'
#python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8 --out_threshold 2.5 --outfile 'wpre_ngn_bs48_ws10_thr8_latent1_batchsize32_epoch50_ot25'
#python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 8.5 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr85_latent1_batchsize32_epoch50_ot2'
#python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --threshold 7.5 --out_threshold 2 --outfile 'wpre_ngn_bs48_ws10_thr75_latent1_batchsize32_epoch50_ot2'

#python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 3 --outfile 'wpre_gn001_bs24_ws10_thr3_latent2_batchsize8_epoch50'
#python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 4 --outfile 'wpre_gn001_bs24_ws10_thr4_latent2_batchsize8_epoch50'
#python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 5 --outfile 'wpre_gn001_bs24_ws10_thr5_latent2_batchsize8_epoch50'
#python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 2 --outfile 'wpre_gn001_bs24_ws10_thr2_latent2_batchsize8_epoch50'
#python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 2.5 --outfile 'wpre_gn001_bs24_ws10_thr25_latent2_batchsize8_epoch50'



#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.99 --latent_dim 2 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr099_latent2_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 8 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent8_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 32 --epoch 10 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize32_epoch10'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 48 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize48_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.95 --latent_dim 1 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr095_latent1_batchsize32_epoch20'

python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --out_threshold 2 --normal_boundary 0.85 --guard_zone 0.97 --p 10 --outfile 'oeco2'
python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 20 --p 10 --outfile 'ff9_sp20_p10'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 300 --out_threshold 2 --threshold 6 --outfile 'cvae10'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 7 --outfile 'cvae9'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 0.1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae8'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 40 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae7'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 30 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae6'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae5'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 2 --kl_weight 1 --epoch 100 --out_threshold 2 --threshold 6 --outfile 'cvae4'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 8 --outfile 'cvae3'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 7 --outfile 'cvae2'
python Continual_VAE/main.py --data './data3/*.npz' --bs 48 --ws 20 --dense_dim 4 --kl_weight 1 --epoch 200 --out_threshold 2 --threshold 6 --outfile 'cvae1'

