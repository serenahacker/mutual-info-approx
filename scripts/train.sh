#!/bin/bash
#SBATCH --output=train-%A.out
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB


basedir="MI-Runs"

run_name=$1
entity=$2

checkpoint_dir="./checkpoint/$USER/$run_name/MI"

model='multi-set-transformer'
dataset='corr'
task='stat/MI'

bs=32
lr="1e-5"
ss1=250
ss2=350
ss_schedule=-1
eval_every=500
save_every=2000
train_steps=100000
val_steps=200
test_steps=500
use_amp=0

weight_decay=0.01
grad_clip=-1

num_blocks=4
num_heads=4
ls=16
hs=32
dropout=0
decoder_layers=1
weight_sharing='none'

n=8

normalize='none'
equi=1
vardim=1

split_inputs=1
decoder_self_attn=0
enc_blocks=4
dec_blocks=1
ln=0
max_rho=0.9

estimate_size=-1
criterion=''
dv_model='encdec'
sample_marg=1
scale='none'
eps="1e-8"

argstring="$run_name $entity --basedir $basedir --checkpoint_dir $checkpoint_dir \
    --model $model --dataset $dataset --task $task --batch_size $bs --lr $lr --set_size $ss1 $ss2 \
    --eval_every $eval_every --save_every $save_every --train_steps $train_steps --val_steps $val_steps \
    --test_steps $test_steps --num_blocks $num_blocks --num_heads $num_heads --latent_size $ls \
    --hidden_size $hs --dropout $dropout --decoder_layers $decoder_layers --weight_sharing $weight_sharing \
    --n $n --normalize $normalize --enc_blocks $enc_blocks --dec_blocks $dec_blocks \
    --max_rho $max_rho --clip $grad_clip --weight_decay $weight_decay --estimate_size $estimate_size \
    --dv_model $dv_model --scale $scale --eps $eps"

if [ $equi -eq 1 ]
then
    argstring="$argstring --equi"
fi
if [ $vardim -eq 1 ]
then
    argstring="$argstring --vardim"
fi
if [ $split_inputs -eq 1 ]
then
    argstring="$argstring --split_inputs"
fi
if [ $decoder_self_attn -eq 1 ]
then
    argstring="$argstring --decoder_self_attn"
fi
if [ $ln -eq 1 ]
then
    argstring="$argstring --layer_norm"
fi
if [ ! -z $criterion ]
then
    argstring="$argstring --criterion $criterion"
fi
if [ $sample_marg -eq 1 ]
then
    argstring="$argstring --sample_marg"
fi


if [ $use_amp -eq 1 ]
then
    argstring="$argstring --use_amp"
fi

python3 main.py $argstring
