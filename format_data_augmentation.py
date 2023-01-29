import argparse
import os
import pickle
import glob

from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="data augmentation script")
    parser.add_argument(
        "--dataset",
        type=str,
        default="biasbios",
        choices=["biasbios", "jigsaw-race"],
        help="datasets",
    )
    parser.add_argument(
        "--aug_type",
        type=str,
        default="backtranslation",
        choices=["EDA", "backtranslation", "clm_insert", "clm_substitute"],
        help="datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="datasets",
    )
    args = parser.parse_args()
    return args

def load_raw_data(args):
    if args.dataset == 'biasbios':
        DATA_DIR = 'data/biasbios-raw/' # ensure the data has been put into the DATA_DIR
        raw_dataset = load_dataset("parquet", 
            data_files={
                f'{args.split}': os.path.join(DATA_DIR, f'biasbios_{args.split}.pq'),
            }
        )
        raw_data = raw_dataset[f'{args.split}']

    elif args.dataset == 'jigsaw-race':
        DATA_DIR = 'data/jigsaw-race/' # ensure the data has been put into the DATA_DIR
        raw_dataset = load_dataset("parquet", 
            data_files={
                f'{args.split}': os.path.join(DATA_DIR, f'jigsaw_race_{args.split}.pq'),
            }
        )
        raw_data = raw_dataset[f'{args.split}']
    else:
        raise NotImplementedError

    return raw_data

def load_augmented_data(args):
    if args.dataset == 'biasbios':
        AUG_DATA_DIR = os.path.join('data/biasbios-raw', 'augmented_text')
    elif args.dataset == 'jigsaw-race':
        AUG_DATA_DIR = os.path.join('data/jigsaw-race/', 'augmented_text')
    else:
        raise NotImplementedError
    
    augmented_text_fn = glob.glob(os.path.join(AUG_DATA_DIR,f'{args.dataset}_{args.split}_{args.aug_type}_*.pkl'))

    text_dict_data = {}
    for pickle_fn in augmented_text_fn:
        with open(pickle_fn, 'rb') as fr:
            text_dict_data_batch = pickle.load(fr)
        text_dict_data.update(text_dict_data_batch)
    
    return text_dict_data

def save_to_output(augmented_data, args):
    if args.dataset == 'biasbios':
        OUT_DIR = 'data/biasbios-raw'
    elif args.dataset == 'jigsaw-race':
        OUT_DIR = 'data/jigsaw-race/'
    else:
        raise NotImplementedError

    dataset = args.dataset.replace("-","_")
    out_fn = os.path.join(OUT_DIR, f"{dataset}_{args.split}_{args.aug_type}.pq")
    print(f"Save output to {out_fn}")
    augmented_data.to_parquet(out_fn)

def main():
    args = parse_args()
    
    # load raw data
    raw_data = load_raw_data(args)
    
    # get augmented_data
    augmented_text_dict_data = load_augmented_data(args)
    augmented_texts = [augmented_text_dict_data[idx] for idx in range(len(augmented_text_dict_data))]

    # assert num. of data are the same
    assert len(raw_data) == len(augmented_text_dict_data) == len(augmented_texts)
    
    # add augmented text to data
    augmented_data = raw_data.add_column(name='augmented_text', column=augmented_texts)

    # # simple validation
    # rand_idx = 11078
    # print("-"*50)
    # print(f"augmented_data[rand_idx] {augmented_data[rand_idx]}")
    # print("-"*50)
    # print(f"augmented_texts[rand_idx] {augmented_texts[rand_idx]}")
    # print("-"*50)

    # save to pq file
    save_to_output(augmented_data, args)

    print()

if __name__ == "__main__":
    main()