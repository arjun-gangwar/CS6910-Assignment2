#!/bin/bash
use_wandb=true
wandb_project=CS6910-Assignment2
wandb_entity=arjungangwar
dataset_path=dataset/inaturalist_12K/
in_dims=224
batch_size=64
data_aug=true
dense_size=512
dropout=0.5
weight_decay=0.005
n_epochs=10
learning_rate=1e-4
freeze_option=0

opts=
if ${use_wandb}; then 
    opts+="--use_wandb "
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
    --dense_size ${dense_size} \
    --dropout ${dropout} \
    --weight_decay ${weight_decay} \
    --n_epochs ${n_epochs} \
    --learning_rate ${learning_rate} \
    --freeze_option ${freeze_option} \
    ${opts}
    
