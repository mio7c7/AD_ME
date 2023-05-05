#!/usr/bin/env bash

python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 20 --p 10 --outfile 'ff9_sp20_p10'
python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 30 --p 10 --outfile 'ff9_sp30_p10'
python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.95 --stabilisation_period 20 --p 10 --outfile 'ff95_sp20_p10'
python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.95 --stabilisation_period 30 --p 10 --outfile 'ff95_sp30_p10'
python OEC/main.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 20 --p 15 --outfile 'ff9_sp20_p15'



