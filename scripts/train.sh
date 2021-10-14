#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109,gpu023

run_name=$1
target=$2
data=$3

if [ $target == "wasserstein" ]
then
    argstring="--target wasserstein --normalize --blur 0.001 --scaling 0.98"
elif [ $target == "kl" ]
then
    argstring="--target kl"
fi

python3 train.py $1 --data $3 --checkpoint_name $SLURM_JOB_ID $argstring --equi