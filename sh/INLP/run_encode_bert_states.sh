#!/bin/bash

dataset="biasbios"
split="train"
model_name_or_path="None"
model_name="None"
output_dir="INLP/data"
addition_params=""

while getopts "d:l:p:n:o:c:" opt; do
    case $opt in
        d ) dataset=$OPTARG;;
        l ) split=$OPTARG;;
        p ) model_name_or_path=$OPTARG;;
        n ) model_name=$OPTARG;;
        o ) output_dir=$OPTARG;;
        c ) addition_params=$OPTARG;;
        *) echo 'Invalid usage'
        exit 1;;
    esac
done

set -x

python INLP/encode_bert_states.py \
        --model_path $model_name_or_path \
        --model_name $model_name \
        --output_dir $output_dir \
        --dataset $dataset \
        --split $split $additional_params