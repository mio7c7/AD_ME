#!/usr/bin/env bash

#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 40 --max_points 1000 --min_batch_size 24 --epsilon 2 --outfile "ws40_mp1000_mbs24_eps2"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 20 --max_points 5000 --min_batch_size 24 --epsilon 2 --outfile "ws20_mp5000_mbs24_eps2"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 20 --max_points 2000 --min_batch_size 24 --epsilon 2 --outfile "ws20_mp2000_mbs24_eps2"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 20 --max_points 1000 --min_batch_size 24 --epsilon 2 --outfile "ws20_mp1000_mbs24_eps2"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 10 --max_points 1000 --min_batch_size 24 --epsilon 2 --outfile "ws10_mp1000_mbs24_eps2"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 30 --max_points 1000 --min_batch_size 24 --epsilon 2 --outfile "ws30_mp1000_mbs24_eps2"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 5 --max_points 1000 --min_batch_size 24 --epsilon 2 --outfile "ws50_mp1000_mbs24_eps2"
#
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 20 --max_points 1000 --min_batch_size 24 --epsilon 0.5 --outfile "ws20_mp1000_mbs24_eps05"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 10 --max_points 1000 --min_batch_size 24 --epsilon 0.5 --outfile "ws10_mp1000_mbs24_eps05"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 30 --max_points 1000 --min_batch_size 24 --epsilon 0.5 --outfile "ws30_mp1000_mbs24_eps05"
#python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 5 --max_points 1000 --min_batch_size 24 --epsilon 0.5 --outfile "ws50_mp1000_mbs24_eps05"

#python LIFEWATCH/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 40 --p 10 --bs 48 --ssa_window 10 -outfile 'ff9_sp40_p10_SSA10_BS48'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 5 --outfile 'wpre_bo20_n10_fo2_th5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 30 --N 10 --fixed_outlier 2 --threshold 2.5 --outfile 'wpre_bo30_n10_fo2_th2.5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 25 --N 8 --fixed_outlier 2 --threshold 2.5 --outfile 'wpre_bo25_n8_fo2_th2.5'
#python main.py --data '../data3/*.npz' --window_size 100 --max_points 100 --min_batch_size 5 --epsilon 1.5 --outfile 'lw02_ws100_mp100_mbs5_eps22'
python main.py --data '../01lr/*.npz' --window_size 100 --max_points 100 --min_batch_size 5 --epsilon 1.5 --outfile 'lw01_ws100_mp100_mbs5_eps15'
python main.py --data '../005lr/*.npz' --window_size 100 --max_points 100 --min_batch_size 5 --epsilon 1.5 --outfile 'lw005_ws100_mp100_mbs5_eps15'

#python main.py --data '../data3/*.npz' --window_size 125 --max_points 80 --min_batch_size 4 --epsilon 1.5 --outfile 'lw02_ws125_mp80_mbs4_eps21'
python main.py --data '../01lr/*.npz' --window_size 125 --max_points 80 --min_batch_size 4 --epsilon 1.5 --outfile 'lw01_ws125_mp80_mbs4_eps15'
python main.py --data '../005lr/*.npz' --window_size 125 --max_points 80 --min_batch_size 4 --epsilon 1.5 --outfile 'lw005_ws125_mp80_mbs4_eps15'

#python main.py --data '../data3/*.npz' --window_size 75 --max_points 133 --min_batch_size 6 --epsilon 1.5 --outfile 'lw02_ws75_mp133_mbs6_eps21'
python main.py --data '../01lr/*.npz' --window_size 75 --max_points 133 --min_batch_size 6 --epsilon 1.5 --outfile 'lw01_ws75_mp133_mbs6_eps15'
python main.py --data '../005lr/*.npz' --window_size 75 --max_points 133 --min_batch_size 6 --epsilon 1.5 --outfile 'lw005_ws75_mp13_mbs6_eps15'

#python main.py --data '../data3/*.npz' --window_size 50 --max_points 200 --min_batch_size 10 --epsilon 1.5 --outfile 'lw02_ws50_mp200_mbs10_eps21'
python main.py --data '../01lr/*.npz' --window_size 50 --max_points 200 --min_batch_size 10 --epsilon 1.5 --outfile 'lw01_ws50_mp200_mbs10_eps15'
python main.py --data '../005lr/*.npz' --window_size 50 --max_points 200 --min_batch_size 10 --epsilon 1.5 --outfile 'lw005_ws50_mp200_mbs10_eps15'
#
#python main.py --data '../data3/*.npz' --window_size 25 --max_points 400 --min_batch_size 20 --epsilon 1.5 --outfile 'lw02_ws25_mp400_mbs20_eps21'
python main.py --data '../01lr/*.npz' --window_size 25 --max_points 400 --min_batch_size 20 --epsilon 1.5 --outfile 'lw01_ws25_mp400_mbs20_eps15'
python main.py --data '../005lr/*.npz' --window_size 25 --max_points 400 --min_batch_size 20 --epsilon 1.5 --outfile 'lw005_ws25_mp400_mbs20_eps15'

#python main.py --data '../data3/*.npz' --window_size 100 --max_points 100 --min_batch_size 5 --epsilon 1.5 --outfile 'lw02_ws100_mp100_mbs5_eps23'
#python main.py --data '../data3/*.npz' --window_size 125 --max_points 80 --min_batch_size 4 --epsilon 1.5 --outfile 'lw02_ws125_mp80_mbs4_eps22'
#python main.py --data '../data3/*.npz' --window_size 75 --max_points 133 --min_batch_size 6 --epsilon 1.5 --outfile 'lw02_ws75_mp133_mbs6_eps22'
#python main.py --data '../data3/*.npz' --window_size 50 --max_points 200 --min_batch_size 10 --epsilon 1.5 --outfile 'lw02_ws50_mp200_mbs10_eps22'
#python main.py --data '../data3/*.npz' --window_size 25 --max_points 400 --min_batch_size 20 --epsilon 1.5 --outfile 'lw02_ws25_mp400_mbs20_eps22'
#
#python main.py --data '../data3/*.npz' --window_size 100 --max_points 100 --min_batch_size 5 --epsilon 1.5 --outfile 'lw02_ws100_mp100_mbs5_eps24'
#python main.py --data '../data3/*.npz' --window_size 125 --max_points 80 --min_batch_size 4 --epsilon 1.5 --outfile 'lw02_ws125_mp80_mbs4_eps23'
#python main.py --data '../data3/*.npz' --window_size 75 --max_points 133 --min_batch_size 6 --epsilon 1.5 --outfile 'lw02_ws75_mp133_mbs6_eps23'
#python main.py --data '../data3/*.npz' --window_size 50 --max_points 200 --min_batch_size 10 --epsilon 1.5 --outfile 'lw02_ws50_mp200_mbs10_eps23'
#python main.py --data '../data3/*.npz' --window_size 25 --max_points 400 --min_batch_size 20 --epsilon 1.5 --outfile 'lw02_ws25_mp400_mbs20_eps23'
#
#
#python main.py --data '../data3/*.npz' --window_size 125 --max_points 80 --min_batch_size 4 --epsilon 1.5 --outfile 'lw02_ws125_mp80_mbs4_eps24'
#python main.py --data '../data3/*.npz' --window_size 75 --max_points 133 --min_batch_size 6 --epsilon 1.5 --outfile 'lw02_ws75_mp133_mbs6_eps24'
#python main.py --data '../data3/*.npz' --window_size 50 --max_points 200 --min_batch_size 10 --epsilon 1.5 --outfile 'lw02_ws50_mp200_mbs10_eps24'
#python main.py --data '../data3/*.npz' --window_size 25 --max_points 400 --min_batch_size 20 --epsilon 1.5 --outfile 'lw02_ws25_mp400_mbs20_eps24'



