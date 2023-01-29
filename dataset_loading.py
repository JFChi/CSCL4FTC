import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import (
    load_dataset,
    set_caching_enabled
)

from transformers import AutoTokenizer

def load_biasbios_for_ce(tokenizer, args, accelerator):
    # Get the datasets
    DATA_DIR = 'data/biasbios-raw/' # ensure the data has been put into the DATA_DIR
    set_caching_enabled(False)
    raw_dataset = load_dataset("parquet", 
        data_files={
            'train': os.path.join(DATA_DIR, 'biasbios_train.pq'),
            'val': os.path.join(DATA_DIR, 'biasbios_val.pq'),
            'test': os.path.join(DATA_DIR, 'biasbios_test.pq'),
        }
    )
    
    # load labels
    def load_label2id(fn):
        label2id = {}
        with open(fn, 'r') as fr:
            for line in fr:
                k, v = line.strip().split('\t')
                label2id[k] = int(v)
        return label2id
    label_to_id = load_label2id(os.path.join(DATA_DIR, "profession2index.txt"))
    protected_group_to_id = load_label2id(os.path.join(DATA_DIR, "gender2index.txt"))
    id_to_label = {id: label for label, id in label_to_id.items()}
    num_labels = len(label_to_id)

    padding = "max_length" if args.pad_to_max_length else False
    
    def biasbios_preprocess_function(examples):
        # Tokenize the texts if untokenized
        texts = examples['hard_text_untokenized']
        result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)

        # add protected_group_labels
        is_male = [int(a=='m') for a in examples['g']]
        is_female = [int(a=='f') for a in examples['g']]
        protected_group_list = [
            is_male, 
            is_female,
        ]
        
        protected_group_labels = list(map(list, zip(*protected_group_list)))
        result["protected_group_labels"] = protected_group_labels
        
        # create labels
        result["labels"] = [label_to_id[label] for label in examples['p']]
        
        return result
    
    if accelerator is not None:
        with accelerator.main_process_first():
            processed_dataset = raw_dataset.map(
                biasbios_preprocess_function,
                batched=True,
                remove_columns=raw_dataset["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    else:
        processed_dataset = raw_dataset.map(
                biasbios_preprocess_function,
                batched=True,
                remove_columns=raw_dataset["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    
    dataset_info = {
        "id_to_label": id_to_label, 
        "label_to_id": label_to_id, 
        "num_labels": num_labels,
        "protected_group_to_id": protected_group_to_id,
    }
    return processed_dataset, dataset_info


def load_biasbios_for_cl(tokenizer, args, accelerator):
    # Get the datasets
    DATA_DIR = 'data/biasbios-raw/' # ensure the data has been put into the DATA_DIR
    train_file = 'biasbios_train.pq' if args.aug_type is None else f'biasbios_train_{args.aug_type}.pq'
    val_file = 'biasbios_val.pq' if args.aug_type is None else f'biasbios_val_{args.aug_type}.pq'
    test_file = 'biasbios_test.pq' if args.aug_type is None else  f'biasbios_test_{args.aug_type}.pq'
    set_caching_enabled(False)
    raw_dataset = load_dataset("parquet", 
        data_files={
            'train': os.path.join(DATA_DIR, train_file),
            'val': os.path.join(DATA_DIR, val_file),
            'test': os.path.join(DATA_DIR, test_file),
        }
    )
    
    # load labels
    def load_label2id(fn):
        label2id = {}
        with open(fn, 'r') as fr:
            for line in fr:
                k, v = line.strip().split('\t')
                label2id[k] = int(v)
        return label2id
    label_to_id = load_label2id(os.path.join(DATA_DIR, "profession2index.txt"))
    protected_group_to_id = load_label2id(os.path.join(DATA_DIR, "gender2index.txt"))
    id_to_label = {id: label for label, id in label_to_id.items()}
    num_labels = len(label_to_id)
    
    padding = "max_length" if args.pad_to_max_length else False
    
    def biasbios_preprocess_function(examples):
        
        total = len(examples['hard_text_untokenized'])
        
        # create different view of input data
        texts = examples['hard_text_untokenized']
        if args.aug_type is None:
            num_aug = 1
            augmented_texts = copy.deepcopy(texts)
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'backtranslation':
            num_aug = 1
            augmented_texts = examples['augmented_text']
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'clm_insert':
            num_aug = 1
            augmented_texts = examples['augmented_text']
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'clm_substitute':
            num_aug = 1
            augmented_texts = examples['augmented_text']
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'EDA':
            augmented_texts = []
            num_aug = len(examples['augmented_text'][0]) # by default, num_aug = 4
            for aug_idx in range(num_aug):
                augmented_texts.extend([aug_text[aug_idx] for aug_text in examples['augmented_text']])
            combined_texts = texts + augmented_texts
        else:
            raise NotImplementedError
        
        # Tokenize the texts if untokenized 
        result = tokenizer(combined_texts, padding=padding, max_length=args.max_length, truncation=True)

        # arange results based on keys and num_aug
        for key in result.keys():
            # result[key] = [[result[key][i], result[key][i+total]] for i in range(total)]
            result[key] = [ [result[key][i+total*j] for j in range(num_aug+1)] for i in range(total) ] 

        # add protected_group_labels
        is_male = [int(a=='m') for a in examples['g']]
        is_female = [int(a=='f') for a in examples['g']]
        protected_group_list = [
            is_male, 
            is_female,
        ]
        
        protected_group_labels = list(map(list, zip(*protected_group_list)))
        result["protected_group_labels"] = protected_group_labels
        
        # create labels
        result["labels"] = [label_to_id[label] for label in examples['p']]
        
        return result
    
    if accelerator is not None:
        with accelerator.main_process_first():
            processed_dataset = raw_dataset.map(
                biasbios_preprocess_function,
                batched=True,
                remove_columns=raw_dataset["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    else:
        processed_dataset = raw_dataset.map(
            biasbios_preprocess_function,
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )
        
    dataset_info = {
        "id_to_label": id_to_label, 
        "label_to_id": label_to_id, 
        "num_labels": num_labels,
        "protected_group_to_id": protected_group_to_id,
    }
    return processed_dataset, dataset_info


def load_jigsaw_race_for_ce(tokenizer, args, accelerator):
    
    # Get the datasets
    DATA_DIR = 'data/jigsaw-race/' # ensure the data has been put into the DATA_DIR
    set_caching_enabled(False)
    raw_dataset = load_dataset("parquet", 
        data_files={
            'train': os.path.join(DATA_DIR, 'jigsaw_race_train.pq'),
            'val': os.path.join(DATA_DIR, 'jigsaw_race_val.pq'),
            'test': os.path.join(DATA_DIR, 'jigsaw_race_test.pq'),
        }
    )
    label_to_id = {
            'non-toxic': 0,
            'toxic': 1,
        }
    protected_group_to_id = {
        'non-black': 0,
        'black': 1,
    }
    id_to_label = {id: label for label, id in label_to_id.items()}
    num_labels = len(label_to_id)

    padding = "max_length" if args.pad_to_max_length else False

    def jigsaw_race_preprocess_function(examples):
        # Tokenize the texts if untokenized
        texts = examples['comment_text']
        result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)

        # add protected_group_labels (one-hot)
        is_non_black = [int(a==0) for a in examples['black']]
        is_black = [int(a==1) for a in examples['black']]
        protected_group_list = [
            is_non_black, 
            is_black,
        ]
        
        protected_group_labels = list(map(list, zip(*protected_group_list)))
        result["protected_group_labels"] = protected_group_labels
        
        # create labels
        result["labels"] = [1 if toxic_score >= 0.5 else 0 for toxic_score in examples['toxicity']]
        
        return result
    
    if accelerator is not None:
        with accelerator.main_process_first():
            processed_dataset = raw_dataset.map(
                jigsaw_race_preprocess_function,
                batched=True,
                remove_columns=raw_dataset["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    else:
        processed_dataset = raw_dataset.map(
            jigsaw_race_preprocess_function,
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    dataset_info = {
        "id_to_label": id_to_label, 
        "label_to_id": label_to_id, 
        "num_labels": num_labels,
        "protected_group_to_id": protected_group_to_id,
    }
    return processed_dataset, dataset_info


def load_jigsaw_race_for_cl(tokenizer, args, accelerator):
        
    # Get the datasets
    DATA_DIR = 'data/jigsaw-race/' # ensure the data has been put into the DATA_DIR
    train_file = 'jigsaw_race_train.pq' if args.aug_type is None else f'jigsaw_race_train_{args.aug_type}.pq'
    val_file = 'jigsaw_race_val.pq' if args.aug_type is None else f'jigsaw_race_val_{args.aug_type}.pq'
    test_file = 'jigsaw_race_test.pq' if args.aug_type is None else  f'jigsaw_race_test_{args.aug_type}.pq'
    set_caching_enabled(False)
    raw_dataset = load_dataset("parquet", 
        data_files={
            'train': os.path.join(DATA_DIR, train_file),
            'val': os.path.join(DATA_DIR, val_file),
            'test': os.path.join(DATA_DIR, test_file),
        }
    )
    label_to_id = {
        'non-toxic': 0,
        'toxic': 1,
    }
    protected_group_to_id = {
        'non-black': 0,
        'black': 1,
    }
    id_to_label = {id: label for label, id in label_to_id.items()}
    num_labels = len(label_to_id)

    padding = "max_length" if args.pad_to_max_length else False

    def jigsaw_race_preprocess_function(examples):
        
        total = len(examples['comment_text'])
        
        # create different view of input data
        texts = examples['comment_text']
        if args.aug_type is None:
            num_aug = 1
            augmented_texts = copy.deepcopy(texts)
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'backtranslation':
            num_aug = 1
            augmented_texts = examples['augmented_text']
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'clm_insert':
            num_aug = 1
            augmented_texts = examples['augmented_text']
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'clm_substitute':
            num_aug = 1
            augmented_texts = examples['augmented_text']
            combined_texts = texts + augmented_texts
        elif args.aug_type == 'EDA':
            augmented_texts = []
            num_aug = len(examples['augmented_text'][0]) # by default, num_aug = 4
            for aug_idx in range(num_aug):
                augmented_texts.extend([aug_text[aug_idx] for aug_text in examples['augmented_text']])
            combined_texts = texts + augmented_texts
        else:
            raise NotImplementedError
        
        # Tokenize the texts if untokenized 
        result = tokenizer(combined_texts, padding=padding, max_length=args.max_length, truncation=True)

        # arange results based on keys and num_aug
        for key in result.keys():
            # result[key] = [[result[key][i], result[key][i+total]] for i in range(total)]
            result[key] = [ [result[key][i+total*j] for j in range(num_aug+1)] for i in range(total) ] 

        # add protected_group_labels (one-hot)
        is_non_black = [int(a==0) for a in examples['black']]
        is_black = [int(a==1) for a in examples['black']]
        protected_group_list = [
            is_non_black, 
            is_black,
        ]
        
        protected_group_labels = list(map(list, zip(*protected_group_list)))
        result["protected_group_labels"] = protected_group_labels
        
        # create labels
        result["labels"] = [1 if toxic_score >= 0.5 else 0 for toxic_score in examples['toxicity']]
        
        return result
    
    if accelerator is not None:
        with accelerator.main_process_first():
            processed_dataset = raw_dataset.map(
                jigsaw_race_preprocess_function,
                batched=True,
                remove_columns=raw_dataset["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    else:
        processed_dataset = raw_dataset.map(
            jigsaw_race_preprocess_function,
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )
        

    dataset_info = {
        "id_to_label": id_to_label, 
        "label_to_id": label_to_id, 
        "num_labels": num_labels,
        "protected_group_to_id": protected_group_to_id,
    }
    return processed_dataset, dataset_info


def load_raw_pq_data(args):
    if args.dataset == 'biasbios':
        DATA_DIR = 'data/biasbios-raw/' # ensure the data has been put into the DATA_DIR
        set_caching_enabled(False)
        raw_dataset = load_dataset("parquet", 
            data_files={
                'train': os.path.join(DATA_DIR, 'biasbios_train.pq'),
                'val': os.path.join(DATA_DIR, 'biasbios_val.pq'),
                'test': os.path.join(DATA_DIR, 'biasbios_test.pq'),
            }
        )
        
        # load labels
        def load_label2id(fn):
            label2id = {}
            with open(fn, 'r') as fr:
                for line in fr:
                    k, v = line.strip().split('\t')
                    label2id[k] = int(v)
            return label2id
        label_to_id = load_label2id(os.path.join(DATA_DIR, "profession2index.txt"))
        protected_group_to_id = load_label2id(os.path.join(DATA_DIR, "gender2index.txt"))
    
    elif args.dataset == 'jigsaw-race':
        DATA_DIR = 'data/jigsaw-race/' # ensure the data has been put into the DATA_DIR
        set_caching_enabled(False)
        raw_dataset = load_dataset("parquet", 
            data_files={
                'train': os.path.join(DATA_DIR, 'jigsaw_race_train.pq'),
                'val': os.path.join(DATA_DIR, 'jigsaw_race_val.pq'),
                'test': os.path.join(DATA_DIR, 'jigsaw_race_test.pq'),
            }
        )
        label_to_id = {
            'non-toxic': 0,
            'toxic': 1,
        }
        protected_group_to_id = {
            'non-black': 0,
            'black': 1,
        }

    else:
        raise NotImplementedError

    dataset_info = {
        "label_to_id": label_to_id, 
        "protected_group_to_id": protected_group_to_id,
    }
    return raw_dataset, dataset_info

def load_text_reps(args):
    path = args.encoded_data_path # os.path.join('INLP', 'data', args.dataset)
    train_fn = f'{args.model_name}_train_cls.npy'
    val_fn = f'{args.model_name}_val_cls.npy'
    test_fn = f'{args.model_name}_test_cls.npy'
    
    x_train = np.load(os.path.join(path, train_fn))
    x_val = np.load(os.path.join(path, val_fn))
    x_test = np.load(os.path.join(path, test_fn))

    return x_train, x_val, x_test

def get_Y_labels(raw_train, raw_val, raw_test, dataset_info, args):
    y2i = dataset_info['label_to_id']
    if args.dataset == 'biasbios':
        y_column_name = 'p'
        y_train = np.array([y2i[entry] for entry in raw_train[y_column_name]])
        y_dev = np.array([y2i[entry] for entry in raw_val[y_column_name]])
        y_test = np.array([y2i[entry] for entry in raw_test[y_column_name]])
    elif args.dataset == 'jigsaw-race':
        y_column_name = 'toxicity'
        y_train = np.array([1 if entry >= 0.5 else 0 for entry in raw_train[y_column_name]])
        y_dev = np.array([1 if entry >= 0.5 else 0 for entry in raw_val[y_column_name]])
        y_test = np.array([1 if entry >= 0.5 else 0 for entry in raw_test[y_column_name]]) 
    else:
        raise NotImplementedError
    
    return y_train, y_dev, y_test

def get_A_labels(raw_train, raw_val, raw_test, dataset_info, args):
    a2i = dataset_info['protected_group_to_id']
    if args.dataset == 'biasbios':
        a_column_name = 'g'
        a_train = np.array([a2i[entry] for entry in raw_train[a_column_name]])
        a_dev = np.array([a2i[entry] for entry in raw_val[a_column_name]])
        a_test = np.array([a2i[entry] for entry in raw_test[a_column_name]])
    elif args.dataset == 'jigsaw-race':
        a_column_name = 'black'
        a_train = np.array([int(entry) for entry in raw_train[a_column_name]])
        a_dev = np.array([int(entry) for entry in raw_val[a_column_name]])
        a_test = np.array([int(entry) for entry in raw_test[a_column_name]])
    else:
        raise NotImplementedError

    return a_train, a_dev, a_test

class TensorDataset(Dataset):
    """
    The medical cost prediction dataset.
    """

    def __init__(self, X, Y, A):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y)
        self.A = torch.from_numpy(A)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data_dict = {
            "input_ids": self.X[idx], 
            "labels": self.Y[idx], 
            "protected_group_labels": self.A[idx],
        }
        return data_dict