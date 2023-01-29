#!/bin/bash
gpu_id=0
seed=0
batch_size=32
addition_params=""
model_name=""
encoded_data_path=""
output_dir="results/INLP/jigsaw-pt"
num_clfs=10

while getopts "g:s:b:p:n:o:l:c:" opt; do
    case $opt in
        g ) gpu_id=$OPTARG;;
        s ) seed=$OPTARG;;
        b ) batch_size=$OPTARG;;
        p ) encoded_data_path=$OPTARG;;
        n ) model_name=$OPTARG;;
        o ) output_dir=$OPTARG;;
        l ) num_clfs=$OPTARG;;
        c ) addition_params=$OPTARG;;
        *) echo 'Invalid usage'
        exit 1;;
    esac
done

dataset="jigsaw-race"
export CUDA_VISIBLE_DEVICES=$gpu_id
set -x

python train_INLP_pt.py \
    --model_name $model_name \
    --dataset $dataset \
    --num_clfs $num_clfs \
    --seed $seed \
    --encoded_data_path $encoded_data_path \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --eval_before_train \
    --output_dir $output_dir $addition_params