#!/bin/bash
use_wandb=true
wandb_project=CS6910-Assignment2
wandb_entity=arjungangwar
dataset_path=dataset/inaturalist_12K/
in_dims=256
batch_norm=true
batch_size=256
n_filters=16
data_aug=false
conv_activation=relu
dense_activation=relu
dense_size=512
filter_size=7,5,5,3,3
filter_org="double"
dropout=0

opts=
if ${use_wandb}; then 
    opts+="--use_wandb "
fi
if ${use_batch_norm}; then 
    opts+="--batch_norm "
fi
if ${data_aug}; then 
    opts+="--data_aug "
fi

python train.py \
    --wandb_project ${wandb_project} \
    --wandb_entity ${wandb_entity} \
    --dataset_path ${dataset_path} \
    --in_dims ${in_dims} \
    --batch_size ${batch_size} \
    --conv_activation ${conv_activation} \
    --dense_activation ${dense_activation} \
    --dense_size ${dense_size} \
    --filter_size ${filter_size} \
    --n_filters ${n_filters} \
    --filter_org ${filter_org} \
    --dropout ${dropout} \
    ${opts}
    
