#!/usr/bin/env bash

#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.95 --latent_dim 2 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr095_latent2_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.95 --latent_dim 4 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr095_latent4_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 16 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize16_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 32 --epoch 40 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize32_epoch40'
#
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.99 --latent_dim 2 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr099_latent2_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 8 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent8_batchsize32_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 32 --epoch 10 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize32_epoch10'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.9 --latent_dim 2 --batch_size 48 --epoch 20 --outfile 'wpre_gn001_bs48_thr09_latent2_batchsize48_epoch20'
#python AE_Detector/main_wpre.py --data './data3/*.npz' --g_noise 0.01 --bs 48 --threshold 0.95 --latent_dim 1 --batch_size 32 --epoch 20 --outfile 'wpre_gn001_bs48_thr095_latent1_batchsize32_epoch20'
#python mainae.py --data '../data3/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae02_gn001_w100_ws100_thr125'
#python mainae.py --data '../01lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae01_gn001_w100_ws100_thr125'
#python mainae.py --data '../005lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae005_gn001_w100_ws100_thr125'
#
#python mainae.py --data '../data3/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1 --outfile 'ae02_gn001_w100_ws100_thr1'
#python mainae.py --data '../01lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1 --outfile 'ae01_gn001_w100_ws100_thr1'
#python mainae.py --data '../005lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 150 --ws 100 --threshold 1 --outfile 'ae005_gn001_w100_ws100_thr1'

python mainae.py --data '../data3/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 100 --ws 50 --threshold 1.25 --outfile 'ae02_gn001_bt5_ws50_thr125'
python mainae.py --data '../01lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 100 --ws 50 --threshold 1.25 --outfile 'ae01_gn001_bt5_ws50_thr125'
python mainae.py --data '../005lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 100 --ws 50 --threshold 1.25 --outfile 'ae005_gn001_bt5_ws50_thr125'

python mainae.py --data '../data3/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 125 --ws 75 --threshold 1.25 --outfile 'ae02_gn001_bt5_ws75_thr125'
python mainae.py --data '../01lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 125 --ws 75 --threshold 1.25 --outfile 'ae01_gn001_bt5_ws75_thr125'
python mainae.py --data '../005lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 125 --ws 175 --threshold 1.25 --outfile 'ae005_gn001_bt5_ws75_thr125'

python mainae.py --data '../data3/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 200 --ws 150 --threshold 1.25 --outfile 'ae02_gn001_bt5_ws150_thr125'
python mainae.py --data '../01lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 200 --ws 150 --threshold 1.25 --outfile 'ae01_gn001_bt5_ws150_thr125'
python mainae.py --data '../005lr/*.npz' --g_noise 0.01 --buffer_ts 500 --bs 200 --ws 150 --threshold 1.25 --outfile 'ae005_gn001_bt5_ws150_thr125'

python mainae.py --data '../data3/*.npz' --g_noise 0.00 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae02_gn000_bt5_ws100_thr1'
python mainae.py --data '../01lr/*.npz' --g_noise 0.00 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae01_gn000_bt5_ws100_thr1'
python mainae.py --data '../005lr/*.npz' --g_noise 0.00 --buffer_ts 500 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae005_gn000_bt5_ws100_thr1'

python mainae.py --data '../data3/*.npz' --g_noise 0.01 --buffer_ts 800 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae02_gn001_bt8_ws100_thr1'
python mainae.py --data '../01lr/*.npz' --g_noise 0.01 --buffer_ts 800 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae01_gn001_bt8_ws100_thr1'
python mainae.py --data '../005lr/*.npz' --g_noise 0.01 --buffer_ts 800 --bs 150 --ws 100 --threshold 1.25 --outfile 'ae005_gn001_bt8_ws100_thr1'