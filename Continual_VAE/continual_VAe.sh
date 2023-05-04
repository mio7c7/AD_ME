#!/usr/bin/env bash

python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 3 --outfile 'wpre_gn001_bs24_ws10_thr3_latent2_batchsize8_epoch50'
python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 4 --outfile 'wpre_gn001_bs24_ws10_thr4_latent2_batchsize8_epoch50'
python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 5 --outfile 'wpre_gn001_bs24_ws10_thr5_latent2_batchsize8_epoch50'
python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 2 --outfile 'wpre_gn001_bs24_ws10_thr2_latent2_batchsize8_epoch50'
python Continual_VAE/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 2.5 --outfile 'wpre_gn001_bs24_ws10_thr25_latent2_batchsize8_epoch50'

#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.99 --latent_dim 2 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr099_latent2_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 8 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent8_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 32 --epoch 10 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize32_epoch10'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 48 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize48_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.95 --latent_dim 1 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr095_latent1_batchsize32_epoch20'



