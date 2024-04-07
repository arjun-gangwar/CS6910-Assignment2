#!/bin/bash
use_wandb=false
wandb_project=CS6910-Assignment2
wandb_entity=arjungangwar
dataset_path=dataset/inaturalist_12K/
in_dims=256
batch_norm=true
batch_size=128
n_filters=32
data_aug=false
conv_activation=gelu
dense_activation=relu
dense_size=1024
filter_size=3,3,5,5,7
filter_org="same"
dropout=0.2
weight_decay=0
n_epochs=15
learning_rate=1e-4

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
    --weight_decay ${weight_decay} \
    --n_epochs ${n_epochs} \
    --learning_rate ${learning_rate} \
    ${opts}
    
