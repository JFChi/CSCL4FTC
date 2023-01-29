import argparse
import os
import pickle
import random
from random import shuffle
import sys
import numpy as np

import torch

from tqdm.auto import tqdm

from datasets import (
    load_dataset,
    set_caching_enabled,
)

import nlpaug.augmenter.word as naw

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the data dataloader.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="datasets",
    )
    parser.add_argument(
        "--idx_start",
        type=int,
        default=0,
        help="example index start",
    )
    parser.add_argument(
        "--idx_end",
        type=int,
        default=100,
        help="example index end",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    args = parser.parse_args()
    # TODO: Sanity checks
    return args

def load_data(args):
    if args.dataset == 'biasbios':
        DATA_DIR = 'data/biasbios-raw/' # ensure the data has been put into the DATA_DIR
        set_caching_enabled(False)
        raw_dataset = load_dataset("parquet", 
            data_files={
                f'{args.split}': os.path.join(DATA_DIR, f'biasbios_{args.split}.pq'),
            }
        )
        raw_data = raw_dataset[f'{args.split}']

    elif args.dataset == 'jigsaw-race':
        DATA_DIR = 'data/jigsaw-race/' # ensure the data has been put into the DATA_DIR
        set_caching_enabled(False)
        raw_dataset = load_dataset("parquet", 
            data_files={
                f'{args.split}': os.path.join(DATA_DIR, f'jigsaw_race_{args.split}.pq'),
            }
        )
        raw_data = raw_dataset[f'{args.split}']
    else:
        raise NotImplementedError

    return raw_data

def extract_texts(data, args):
    if args.dataset == 'biasbios':
        texts = data['hard_text_untokenized']
    elif args.dataset == 'jigsaw-race':
        texts = data['comment_text']
    else:
        raise NotImplementedError

    return texts

def save_to_output(augmented_texts, args):
    if args.dataset == 'biasbios':
        OUT_DIR = os.path.join('data/biasbios-raw', 'augmented_text')
    elif args.dataset == 'jigsaw-race':
        OUT_DIR = os.path.join('data/jigsaw-race/', 'augmented_text')
    else:
        raise NotImplementedError

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    out_dict = {}
    for idx, text in enumerate(augmented_texts):
        out_dict[args.idx_start+idx] = text

    out_fn = f'{args.dataset}_{args.split}_{args.aug_type}_{args.idx_start}_{args.idx_end}.pkl'
    out_fn = os.path.join(OUT_DIR, out_fn)
    print(f"saving augmented data to {out_fn}")
    with open(out_fn, 'wb') as fw:
        pickle.dump(out_dict, fw)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def back_translation_augmentation(texts, args):
    if args.dataset == 'biasbios':
        max_length = 192
    elif args.dataset == 'jigsaw-race':
        max_length = 242
    else:
        raise NotImplementedError
    
    # init augmenter
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en',
        device=device,
        batch_size=args.batch_size,
        max_length=max_length,
    )

    augmented_texts = []
    progress_bar = tqdm(range(len(texts)))
    for texts_batch in batch(texts, args.batch_size):
        augmented_texts_batch = back_translation_aug.augment(texts_batch)
        augmented_texts.extend(augmented_texts_batch)
        progress_bar.update(len(texts_batch))

    return augmented_texts

def EDA_augmentation(texts, args):

    #################################################################
    ############# add nessary components for EDA  ###################
    #################################################################
    #stop words list
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
                'ours', 'ourselves', 'you', 'your', 'yours', 
                'yourself', 'yourselves', 'he', 'him', 'his', 
                'himself', 'she', 'her', 'hers', 'herself', 
                'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 
                'whom', 'this', 'that', 'these', 'those', 'am', 
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did',
                'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                'because', 'as', 'until', 'while', 'of', 'at', 
                'by', 'for', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 
                'above', 'below', 'to', 'from', 'up', 'down', 'in',
                'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once', 'here', 'there', 'when', 
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                'very', 's', 't', 'can', 'will', 'just', 'don', 
                'should', 'now', '']

    #cleaning up text
    import re
    def get_only_chars(line):

        clean_line = ""

        line = line.replace("’", "")
        line = line.replace("'", "")
        line = line.replace("-", " ") #replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = line.lower()

        for char in line:
            if char in 'qwertyuiopasdfghjklzxcvbnm ':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    ########################################################################
    # Synonym replacement
    # Replace n words in the sentence with synonyms from wordnet
    ########################################################################

    #for the first time you use wordnet
    #import nltk
    #nltk.download('wordnet')
    from nltk.corpus import wordnet 

    def synonym_replacement(words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                #print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n: #only replace up to n words
                break

        #this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    ########################################################################
    # Random deletion
    # Randomly delete words from the sentence with probability p
    ########################################################################

    def random_deletion(words, p):

        #obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words

    ########################################################################
    # Random swap
    # Randomly swap two words in the sentence n times
    ########################################################################

    def random_swap(words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = swap_word(new_words)
        return new_words

    def swap_word(new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words

    ########################################################################
    # Random insertion
    # Randomly insert n words into the sentence
    ########################################################################

    def random_insertion(words, n):
        new_words = words.copy()
        for _ in range(n):
            add_word(new_words)
        return new_words

    def add_word(new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    ########################################################################
    # main data augmentation function
    ########################################################################

    def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']
        num_words = len(words)
        
        augmented_sentences = []
        num_new_per_technique = int(num_aug/4)+1

        #sr
        if (alpha_sr > 0):
            n_sr = max(1, int(alpha_sr*num_words))
            for _ in range(num_new_per_technique):
                a_words = synonym_replacement(words, n_sr)
                augmented_sentences.append(' '.join(a_words))

        #ri
        if (alpha_ri > 0):
            n_ri = max(1, int(alpha_ri*num_words))
            for _ in range(num_new_per_technique):
                a_words = random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))

        #rs
        if (alpha_rs > 0):
            n_rs = max(1, int(alpha_rs*num_words))
            for _ in range(num_new_per_technique):
                a_words = random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))

        #rd
        if (p_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = random_deletion(words, p_rd)
                augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)

        #trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        return augmented_sentences

    #################################################################
    ################### begin EDA augmentation ######################
    #################################################################


    progress_bar = tqdm(range(len(texts)))
    augmented_texts = []
    for orig_text in texts:
        augmented_text_N = eda(orig_text, num_aug=4)
        augmented_texts.append(augmented_text_N)
        progress_bar.update(1)

    return augmented_texts

def contextual_lm_insertion_augmentation(texts, args):
    if args.dataset == 'biasbios':
        max_length = 192
    elif args.dataset == 'jigsaw-race':
        max_length = 242
    else:
        raise NotImplementedError
    
    # contextual word embedding (insertion)
    aug = naw.ContextualWordEmbsAug(
        model_path='roberta-base', 
        action="insert",
        aug_p=0.1,
        device=device,
        batch_size=args.batch_size,
    )
    augmented_texts = []
    progress_bar = tqdm(range(len(texts)))
    for texts_batch in batch(texts, args.batch_size):
        augmented_texts_batch = aug.augment(texts_batch)
        augmented_texts.extend(augmented_texts_batch)
        progress_bar.update(len(texts_batch))

    return augmented_texts

def contextual_lm_substitute_augmentation(texts, args):
    if args.dataset == 'biasbios':
        max_length = 192
    elif args.dataset == 'jigsaw-race':
        max_length = 242
    else:
        raise NotImplementedError
    
    # contextual word embedding (substitute)
    aug = naw.ContextualWordEmbsAug(
        model_path='roberta-base', 
        action="substitute",
        aug_p=0.1,
        device=device,
        batch_size=args.batch_size,
    )
    augmented_texts = []
    progress_bar = tqdm(range(len(texts)))
    for texts_batch in batch(texts, args.batch_size):
        augmented_texts_batch = aug.augment(texts_batch)
        augmented_texts.extend(augmented_texts_batch)
        progress_bar.update(len(texts_batch))

    return augmented_texts

def main():
    args = parse_args()

    # set seed and begin augmentation
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # load data
    raw_data = load_data(args)

    print(raw_data)
    # extract texts from raw data
    texts = extract_texts(raw_data, args)

    # select range to augment
    args.idx_start = max(0, args.idx_start)
    args.idx_end = min(len(texts), args.idx_end)

    print(f"Augmenting text with range {args.idx_start, args.idx_end}")
    texts = texts[args.idx_start:args.idx_end]

    # TODO augment
    if args.aug_type == 'backtranslation':
        augmented_texts = back_translation_augmentation(texts, args)
    elif args.aug_type == 'EDA':
        augmented_texts = EDA_augmentation(texts, args)
    elif args.aug_type == 'clm_insert':
        augmented_texts = contextual_lm_insertion_augmentation(texts, args)
    elif args.aug_type == 'clm_substitute':
        augmented_texts = contextual_lm_substitute_augmentation(texts, args)
    else:
        raise NotImplementedError

    # save to augment output
    save_to_output(augmented_texts, args)

if __name__ == "__main__":
    main()
    