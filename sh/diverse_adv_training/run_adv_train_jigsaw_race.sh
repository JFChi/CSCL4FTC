#!/bin/bash
set -x
gpu_id=0
seed=0
batch_size=16
model_path=bert-base-uncased
addition_params=""
out_path="results/test_adv_train"

lambda_adv=0.0
lambda_diff=100
n_discriminators=3

while getopts "g:s:b:w:p:t:c:o:l:r:n:" opt; do
    case $opt in
        g ) gpu_id=$OPTARG;;
        s ) seed=$OPTARG;;
        b ) batch_size=$OPTARG;;
        p ) model_path=$OPTARG;;
        c ) addition_params=$OPTARG;;
        o ) out_path=$OPTARG;;
        l ) lambda_adv=$OPTARG;;
        r ) lambda_diff=$OPTARG;;
        n ) n_discriminators=$OPTARG;;
        *) echo 'Invalid usage'
        exit 1;;
    esac
done

dataset="jigsaw-race"
max_length=242

export CUDA_VISIBLE_DEVICES=$gpu_id

python train_diverse_adv.py \
  --model_name_or_path $model_path \
  --max_length $max_length \
  --per_device_train_batch_size $batch_size \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --adv_training_epochs 10 \
  --eval_before_train \
  --output_dir $out_path \
  --lambda_adv $lambda_adv \
  --lambda_diff $lambda_diff \
  --n_discriminators $n_discriminators \
  --seed $seed \
  --dataset $dataset $addition_params