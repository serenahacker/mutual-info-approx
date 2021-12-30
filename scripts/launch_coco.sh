#!/bin/bash

n_runs=3

batch_size=6
num_blocks=1
latent_size=512
hidden_size=1024
lr="1e-5"
basedir="final-runs"
run_name=$1
steps=16000
set_size1=3
set_size2=10

for (( i = 0 ; i < $n_runs ; i++ ))
do
    sbatch scripts/train_coco.sh "${run_name}/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "csab" $set_size1 $set_size2
    sbatch scripts/train_coco.sh "${run_name}_naive/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "naive" $set_size1 $set_size2
    sbatch scripts/train_coco.sh "${run_name}_pine/${i}" $batch_size $latent_size $hidden_size $lr $num_blocks $steps "pine" $set_size1 $set_size2
done
