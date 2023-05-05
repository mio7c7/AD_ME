#!/usr/bin/env bash

#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 4.5 --bs 48 --ssa_window 10 --outfile 'wpre_bo20_n10_fo2_th4.5_SSA10_BS48'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 4 --bs 48 --ssa_window 10 --outfile 'wpre_bo20_n10_fo2_th4'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 3 --bs 48 --ssa_window 10 --outfile 'wpre_bo20_n10_fo2_th3_SSA10_BS48'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 30 --N 10 --fixed_outlier 2 --threshold 4.5 --outfile 'wpre_bo30_n10_fo2_th4.5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 25 --N 8 --fixed_outlier 2 --threshold 4.5 --outfile 'wpre_bo25_n8_fo2_th4.5'
#
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 2.5 --outfile 'wpre_bo20_n10_fo2_th2.5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 2 --bs 48 --ssa_window 10 --outfile 'wpre_bo20_n10_fo2_th2_SSA10_BS48'
python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 20 --p 10 --bs 48 --ssa_window 10 -outfile 'ff9_sp20_p10_SSA10_BS48'
python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 40 --p 10 --bs 48 --ssa_window 10 -outfile 'ff9_sp40_p10_SSA10_BS48'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 2 --threshold 5 --outfile 'wpre_bo20_n10_fo2_th5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 30 --N 10 --fixed_outlier 2 --threshold 2.5 --outfile 'wpre_bo30_n10_fo2_th2.5'
#python MStatistics/main_wpre.py --data './data3/*.npz' --bo 25 --N 8 --fixed_outlier 2 --threshold 2.5 --outfile 'wpre_bo25_n8_fo2_th2.5'



