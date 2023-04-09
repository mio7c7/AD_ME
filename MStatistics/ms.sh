#!/usr/bin/env bash

python MStatistics/main.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 1 --threshold 4.5 --outfile 'bo20_n10_fo1_th4.5'
python MStatistics/main.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 1 --threshold 4 --outfile 'bo20_n10_fo1_th4'
python MStatistics/main.py --data './data3/*.npz' --bo 20 --N 10 --fixed_outlier 1 --threshold 3 --outfile 'bo20_n10_fo1_th3'
python MStatistics/main.py --data './data3/*.npz' --bo 30 --N 10 --fixed_outlier 1 --threshold 4.5 --outfile 'bo30_n10_fo1_th4.5'
python MStatistics/main.py --data './data3/*.npz' --bo 25 --N 8 --fixed_outlier 1 --threshold 4.5 --outfile 'bo25_n8_fo1_th4.5'



