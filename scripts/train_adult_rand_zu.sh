#!/bin/bash
#SBATCH --output=train-%A.out
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB


basedir="MI-Runs"

run_name=$1
entity=$2

checkpoint_dir="./checkpoint/$USER/$run_name/DV-MI/${SLURM_JOB_ID}"

# stat/MI for supervised, stat/DV-MI for unsupervised
task='stat/DV-MI'

###     data parameters
# 'corr' for synthetic gaussian data, 'adult' for adult dataset
dataset='adult-rand-zu-2'
# dimensionality of the data
n=10
# correlations go from -max_rho to +max_rho for 'corr' dataset
max_rho=0.9

###     general training parameters
bs=32
# lr 1e-4 for supervised or 1e-5 for unsupervised
lr="1e-6"
# total number of training steps as well as how often to evaluate/save and how many steps to perform for evaluation
train_steps=100000
eval_every=500
save_every=2000
val_steps=200
test_steps=500
# whether or not to use amp acceleration
use_amp=0

## set size parameters
# set sizes go from ss1 to ss2
ss1=250
ss2=350
ss_schedule=-1


weight_decay=0
grad_clip=-1


###     general model parameters
ls=32
hs=64
num_heads=4
decoder_layers=1
dropout=0
ln=0
# 'none' or 'whiten'
normalize='none'

weight_sharing='none'

# these parameters control the dimension equivariance. 
# they should usually be set to the same value: 1 to use it or 0 to not use it
equi=0
vardim=0

###     supervised parameters
model='multi-set-transformer'
num_blocks=4

###     unsupervised parameters
dv_model='encdec'
enc_blocks=4
dec_blocks=1
eps="1e-6"

scale='none'
sample_marg=0
estimate_size=-1
split_inputs=1
decoder_self_attn=0

criterion=''


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
