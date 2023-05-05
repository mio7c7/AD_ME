#!/bin/bash
#SBATCH --cpus-per-task=1
###SBATCH --nodelist=gpuserver1.csit.local
###SBATCH --exclude=node[1-5]
#SBATCH -J akari
#SBATCH --nodes=1
#SBATCH --partition=SCT
#SBATCH --mem=16G
#SBATCH --error=err.txt

#MAINBATCHCOMMANDS
#
#echo"RUNNING$@"
echo"python $@"
python "$@"

#cd-
