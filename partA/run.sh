#!/bin/bash
use_wandb=false
dataset_path=dataset/inaturalist_12K/

python train.py \
    --use_wandb ${use_wandb} \
    --dataset_path ${dataset_path}