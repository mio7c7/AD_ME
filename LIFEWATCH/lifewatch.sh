#!/usr/bin/env bash

python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 20 --max_points 1000 --min_batch_size 24 --epsilon 1 --outfile "ws20_mp1000_mbs24_eps1"
python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 10 --max_points 1000 --min_batch_size 24 --epsilon 1 --outfile "ws10_mp1000_mbs24_eps1"
python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 30 --max_points 1000 --min_batch_size 24 --epsilon 1 --outfile "ws30_mp1000_mbs24_eps1"
python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 5 --max_points 1000 --min_batch_size 24 --epsilon 1 --outfile "ws50_mp1000_mbs24_eps1"

python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 20 --max_points 1000 --min_batch_size 24 --epsilon 1.5 --outfile "ws20_mp1000_mbs24_eps15"
python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 10 --max_points 1000 --min_batch_size 24 --epsilon 1.5 --outfile "ws10_mp1000_mbs24_eps15"
python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 30 --max_points 1000 --min_batch_size 24 --epsilon 1.5 --outfile "ws30_mp1000_mbs24_eps15"
python LIFEWATCH/main_wpre.py --data './data3/*.npz' --window_size 5 --max_points 1000 --min_batch_size 24 --epsilon 1.5 --outfile "ws50_mp1000_mbs24_eps15"

#python LIFEWATCH/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 40 --p 10 --bs 48 --ssa_window 10 -outfile 'ff9_sp40_p10_SSA10_BS48'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 5 --outfile 'wpre_bo20_n10_fo2_th5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 30 --N 10 --fixed_outlier 2 --threshold 2.5 --outfile 'wpre_bo30_n10_fo2_th2.5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 25 --N 8 --fixed_outlier 2 --threshold 2.5 --outfile 'wpre_bo25_n8_fo2_th2.5'



