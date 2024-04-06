#!/bin/bash
use_wandb=false
dataset_path=dataset/inaturalist_12K/
in_dims=256
use_batch_norm=false
batch_size=64
n_filter=256
filter_org=half

opts=
if ${use_wandb}; then 
    opts+="--use_wandb "
fi
if ${use_batch_norm}; then 
    opts+="--use_batch_norm "
fi 

python train.py \
    --dataset_path ${dataset_path} \
    --in_dims ${in_dims} \
    --batch_size ${batch_size} \
    ${opts}
    
