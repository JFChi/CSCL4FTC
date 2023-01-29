#!/bin/bash
set -x
gpu_id=0
seed=0
batch_size=32
addition_params=""

while getopts "g:s:b:w:p:t:c:" opt; do
    case $opt in
        g ) gpu_id=$OPTARG;;
        s ) seed=$OPTARG;;
        b ) batch_size=$OPTARG;;
        p ) model_path=$OPTARG;;
        c ) addition_params=$OPTARG;;
        *) echo 'Invalid usage'
        exit 1;;
    esac
done

dataset="jigsaw-race"
max_length=242

export CUDA_VISIBLE_DEVICES=$gpu_id
model_type=bert

python finetune_CE.py \
    --model_name_or_path $model_path \
    --model_type $model_type \
    --max_length $max_length \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 7 \
    --eval_before_train \
    --output_dir ${model_path}/ce_finetune \
    --seed $seed \
    --dataset $dataset $addition_params