
'''
Code modified from https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/data/encode_bert_states.py
'''

import numpy as np
import os
import argparse
import pickle

import torch

from transformers import BertModel, BertTokenizer, BertConfig
from datasets import load_dataset
from tqdm import tqdm

def encode_text(model, data):
    """
    encode the text
    :param model: encoding model
    :param data: data
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """

    all_data_cls = []
    
    for row in tqdm(data):
        input_ids = torch.tensor(row.data['input_ids']).unsqueeze(0)
        token_type_ids = torch.tensor(row.data['token_type_ids']).unsqueeze(0)
        attention_mask = torch.tensor(row.data['attention_mask']).unsqueeze(0)
        with torch.no_grad():
            last_hidden_states = model(
                input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask
            )[0]
            all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
    return np.array(all_data_cls)
    
def parse_args():
    parser = argparse.ArgumentParser(description="data augmentation script")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="pretrain model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="biasbios",
        choices=["biasbios", "jigsaw-race"],
        help="datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="split",
    )
    args = parser.parse_args()
    
    # Sanity checks
    if args.model_path != "bert-base-uncased":
        assert os.path.exists(args.model_path)
        assert args.model_name is not None
    else:
        args.model_name = "bert-base-uncased"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    return args

def load_data(args):
    if args.dataset == 'biasbios':
        DATA_DIR = 'data/biasbios-raw/' # ensure the data has been put into the DATA_DIR
        raw_dataset = load_dataset("parquet", 
            data_files={
                f'{args.split}': os.path.join(DATA_DIR, f'biasbios_{args.split}.pq'),
            }
        )
        raw_data = raw_dataset[f'{args.split}']
        text_column = 'hard_text_untokenized'
    elif args.dataset == 'jigsaw-race':
        DATA_DIR = 'data/jigsaw-race/' # ensure the data has been put into the DATA_DIR
        raw_dataset = load_dataset("parquet", 
            data_files={
                f'{args.split}': os.path.join(DATA_DIR, f'jigsaw_race_{args.split}.pq'),
            }
        )
        raw_data = raw_dataset[f'{args.split}']
        text_column = 'comment_text'
    else:
        raise NotImplementedError

    return raw_data, text_column

def load_lm(args):
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    pretrained_weights = args.model_path
    config = BertConfig.from_pretrained(pretrained_weights)
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights)
    return model, tokenizer, config


def tokenize(tokenizer, data, text_column):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :param text_column: text column name
    :return: a list of the entire tokenized data
    """
    
    tokenized_data = []
    for text in tqdm(data[text_column]):
        tokenized_data_item = tokenizer(
            text,
            padding=False,
            max_length=512, 
            truncation=True, 
            add_special_tokens=True, 
        )
        tokenized_data.append(tokenized_data_item)
    return tokenized_data
    
if __name__ == '__main__':
    args = parse_args()
    
    out_dir = args.output_dir
    split = args.split

    model, tokenizer, config = load_lm(args)
    pq_data, text_column = load_data(args)
    tokens = tokenize(tokenizer, pq_data, text_column)
    cls_data = encode_text(model, tokens)

    # save bert cls file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fn = os.path.join(out_dir, f"{args.model_name}_{args.split}_cls.npy")
    np.save(out_fn, cls_data)
