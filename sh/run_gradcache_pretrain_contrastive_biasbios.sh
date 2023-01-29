#!/bin/bash
set -x

gpu_id=0
seed=0
batch_size=128
aux_loss_weight=0.0
model_path="results"
temperature=2.0
addition_params=""

while getopts "g:s:b:w:p:t:c:" opt; do
    case $opt in
        g ) gpu_id=$OPTARG;;
        s ) seed=$OPTARG;;
        b ) batch_size=$OPTARG;;
        w ) aux_loss_weight=$OPTARG;;
        p ) model_path=$OPTARG;;
        t ) temperature=$OPTARG;;
        c ) addition_params=$OPTARG;;
        *) echo 'Invalid usage'
        exit 1;;
    esac
done

dataset="biasbios"
max_length=192

export CUDA_VISIBLE_DEVICES=$gpu_id

num_train_epochs=15

python pretrain_cl_gradcache.py \
  --model_name_or_path bert-base-uncased \
  --max_length $max_length \
  --per_device_train_batch_size ${batch_size} \
  --per_device_eval_batch_size ${batch_size} \
  --learning_rate 2e-5 \
  --num_train_epochs ${num_train_epochs} \
  --eval_before_train \
  --output_dir $model_path \
  --save_model \
  --gradcache_chunk_size 16 \
  --seed $seed \
  --aux_loss_weight $aux_loss_weight \
  --temperature $temperature \
  --dataset $dataset $addition_params