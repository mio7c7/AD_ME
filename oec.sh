#!/usr/bin/env bash

python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.9 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '02_ff9_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.8 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '02_ff8_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.7 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '02_ff7_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.6 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '02_ff6_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './data3/*.npz' --forgetting_factor 0.5 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '02_ff5_995_sp25_p20_cs2'

python OEC_otfilteronly/main_wpre.py --data './01lr/*.npz' --forgetting_factor 0.9 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '01_ff9_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './01lr/*.npz' --forgetting_factor 0.8 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '01_ff8_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './01lr/*.npz' --forgetting_factor 0.7 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '01_ff7_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './01lr/*.npz' --forgetting_factor 0.6 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '01_ff6_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './01lr/*.npz' --forgetting_factor 0.5 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '01_ff5_995_sp25_p20_cs2'

python OEC_otfilteronly/main_wpre.py --data './005lr/*.npz' --forgetting_factor 0.9 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '005_ff9_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './005lr/*.npz' --forgetting_factor 0.8 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '005_ff8_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './005lr/*.npz' --forgetting_factor 0.7 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '005_ff7_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './005lr/*.npz' --forgetting_factor 0.6 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '005_ff6_995_sp25_p20_cs2'
python OEC_otfilteronly/main_wpre.py --data './005lr/*.npz' --forgetting_factor 0.5 --stabilisation_period 25 --normal_boundary 0.9 --guard_zone 0.95 --p 20 --cs 2 --outfile '005_ff5_995_sp25_p20_cs2'

