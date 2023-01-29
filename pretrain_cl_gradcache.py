#!/usr/bin/env python
# coding=utf-8

# code adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py 
""" fair contrastive pretraining with no trainer."""

import argparse
import sys
import logging
import math
import os
import random
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Union,
    Any,
    Dict,
    Optional,
    List,
)

import torch
import torch.nn.functional as F
import torch.distributed as dist
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from models.bert.modeling_bert import BertForCLGradCache
from models.cl_grad_cache import GradCache

from utils import set_cl_eval_mode

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1000, help="Number of steps for logging the train loss."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    # new arguments comes here
    parser.add_argument(
        "--eval_before_train",
        action="store_true",
        help="evaluation before training",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="save_model or not",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jigsaw",
        choices=["biasbios", "jigsaw-race"],
        help="datasets",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate.",
    )
    parser.add_argument(
        "--pooler_type",
        type=str,
        default="cls",
        help="pooler type of contrastive learning",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature",
    )
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        default=1.0,
        help="loss_2 weight.",
    )
    parser.add_argument(
        "--gradcache_chunk_size",
        type=int,
        default=16,
        help="chunk size in grad cache",
    )
    parser.add_argument(
        "--aug_type",
        type=str,
        default=None,
        choices=[None, "backtranslation", "EDA", "clm_insert", "clm_substitute"],
        help="datasets",
    )
    
    args = parser.parse_args()

    # Sanity checks
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        assert args.model_name_or_path in ['bert-base-uncased']
    return args

def eval_model(model, gradcache, eval_dataloader, accelerator):
    set_cl_eval_mode(model)

    with torch.no_grad():
        completed_eval_steps = 0
        eval_loss_val = torch.tensor(0.0).to(accelerator.device)
        eval_loss_1_val = torch.tensor(0.0).to(accelerator.device)
        eval_loss_2_val = torch.tensor(0.0).to(accelerator.device)
        for eval_step, batch in enumerate(eval_dataloader):
            loss_dict = gradcache.step(batch)
            eval_loss_val += accelerator.gather(loss_dict['loss']).mean()
            eval_loss_1_val += accelerator.gather(loss_dict['loss_1']).mean()
            eval_loss_2_val += accelerator.gather(loss_dict['loss_2']).mean()
            completed_eval_steps += 1

        # log eval loss
        log_info: Dict[str, float] = {}
        eval_loss_val_scalar = eval_loss_val.item()
        eval_loss_1_val_scalar = eval_loss_1_val.item()
        eval_loss_2_val_scalar = eval_loss_2_val.item()
        
        # compute loss for each step ang log
        log_info['eval_overall_loss'] = round(eval_loss_val_scalar / completed_eval_steps, 6)
        log_info['eval_loss_1'] = round(eval_loss_1_val_scalar / completed_eval_steps, 6)
        log_info['eval_loss_2'] = round(eval_loss_2_val_scalar / completed_eval_steps, 6)
        
        # log it
        logger.info(f"Eval loss: {log_info}")
        
        return eval_loss_val / completed_eval_steps

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, f"train_{args.seed}.log")),
            logging.StreamHandler(),
        ],
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        
    # load datasets and tokenizer, and preprocess datasets
    if args.dataset == 'biasbios':
        from dataset_loading import load_biasbios_for_cl
        
        # load tokenizer and preprocessing the datasets    
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=not args.use_slow_tokenizer
        )
        
        processed_dataset, dataset_info  = load_biasbios_for_cl(tokenizer, args, accelerator)
        id_to_label = dataset_info["id_to_label"]
        label_to_id = dataset_info["label_to_id"]
        num_labels = dataset_info["num_labels"]
        
        train_dataset = processed_dataset["train"]
        val_dataset = processed_dataset["val"]
        test_dataset = processed_dataset["test"]
        
    elif args.dataset == 'jigsaw-race':
        from dataset_loading import load_jigsaw_race_for_cl
        
        # load tokenizer and preprocessing the datasets    
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=not args.use_slow_tokenizer
        )
        
        processed_dataset, dataset_info = load_jigsaw_race_for_cl(tokenizer, args, accelerator)
        
        id_to_label = dataset_info["id_to_label"]
        label_to_id = dataset_info["label_to_id"]
        num_labels = dataset_info["num_labels"]
        
        train_dataset = processed_dataset["train"]
        val_dataset = processed_dataset["val"]
        test_dataset = processed_dataset["test"]
    
    else:
        raise NotImplementedError
        
    # Load pretrained model and config based on datasets
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, 
        num_labels=num_labels
    )
    # change model configuration if needed (e.g., dropout, etc)
    config.attention_probs_dropout_prob = args.dropout
    config.hidden_dropout_prob = args.dropout
    
    if args.model_name_or_path == 'bert-base-uncased':
        model = BertForCLGradCache.from_pretrained(
            args.model_name_or_path,
            config=config,
            model_args=args,
        )
    else:
        raise NotImplementedError
    
    model.config.label2id = label_to_id
    model.config.id2label = id_to_label

    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 2):
    # # # Log a few random samples from the valid set:
    # for index in [33]:
    #     logger.info(f"Sample {index} of the validation set: {val_dataset[index]}.")
    
    # Log some parameters here!
    logger.info(f"parser args : {vars(args)}")

    # Data collator
    @dataclass
    class CLDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = (8 if accelerator.use_fp16 else None)

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids'] # keys with data augmentation elements
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            # have two views of data
            if num_sent == 2:
                for feature in features:
                    for i in range(num_sent):
                        flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})
            # have more than two views of data
            else:
                for feature in features:
                    sent_idx = 0
                    # add orginal sentence
                    flat_features.append({k: feature[k][sent_idx] if k in special_keys else feature[k] for k in feature})
                    # choose random random augmented index other than the original sentence
                    sent_idx = random.randint(1, num_sent-1)
                    flat_features.append({k: feature[k][sent_idx] if k in special_keys else feature[k] for k in feature})

            assert len(flat_features) == 2 * bs
            
            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch = {k: batch[k].view(bs, 2, -1) if k in special_keys else batch[k].view(bs, 2, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]
            if "protected_group_label" in batch:
                batch["protected_group_labels"] = batch["protected_group_label"]
                del batch["protected_group_labels"]

            return batch

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = CLDataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.gradient_accumulation_steps != 1:
        raise ValueError("In the current implementation of CL, "
         "we only support gradient_accumulation_steps equal to one. "
         "Otherwise it is hard to calculate the exact batch size"
        )

    # initialize gradcache
    gc = GradCache(
        model=model,
        args=args,
        accelerator=accelerator
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    if args.eval_before_train:
        eval_loss_val = eval_model(model, gc, eval_dataloader, accelerator)

    running_loss_val = torch.tensor(0.0).to(accelerator.device)
    running_loss_1_val = torch.tensor(0.0).to(accelerator.device)
    running_loss_2_val = torch.tensor(0.0).to(accelerator.device)
    best_eval_loss_val = eval_loss_val if args.eval_before_train else torch.tensor(float('inf')).to(accelerator.device)
    best_epoch = 0 # epoch achieve the best loss
    globalstep_last_logged = 0
    
    for epoch in range(args.num_train_epochs):
        # train step
        model.train()
        for step, batch in enumerate(train_dataloader):

            loss_dict = gc.cache_step(batch)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            # accumulate value for logging
            # NOTE: loss in every process is the same due to allreduce
            running_loss_val += loss_dict['loss']
            running_loss_1_val += loss_dict['loss_1']
            running_loss_2_val += loss_dict['loss_2']
            
            # log loss for each interval 
            if (completed_steps-globalstep_last_logged) % args.logging_steps == 0:
                
                log_info: Dict[str, float] = {}
                
                running_loss_val_scalar = running_loss_val.item()
                running_loss_1_val_scalar = running_loss_1_val.item()
                running_loss_2_val_scalar = running_loss_2_val.item()

                # reset tr_loss to zero
                running_loss_val -= running_loss_val
                running_loss_1_val -= running_loss_1_val
                running_loss_2_val -= running_loss_2_val

                # compute loss for each step ang log
                log_info['overall_loss'] = round(running_loss_val_scalar / (completed_steps - globalstep_last_logged), 6)
                log_info['loss_1'] = round(running_loss_1_val_scalar / (completed_steps - globalstep_last_logged), 6)
                log_info['loss_2'] = round(running_loss_2_val_scalar / (completed_steps - globalstep_last_logged), 6)

                # update globalstep_last_logged
                globalstep_last_logged = completed_steps
                
                # log it
                logger.info(f"logging training loss: {log_info}")
            
            if completed_steps >= args.max_train_steps:
                break
        
        # eval model on val set after each epoch
        eval_loss_val = eval_model(model, gc, eval_dataloader, accelerator)

        #  save model after each epoch if achieve best eval_loss
        if args.output_dir is not None and args.save_model \
            and eval_loss_val.item() <= best_eval_loss_val.item():
            
            logger.info(
                f"After running {epoch+1} epoch(s), saving model. "
                f"Achieving best loss: {eval_loss_val.item()}, "
                f"and previous best loss {best_eval_loss_val.item()}"
            )
            
            # reset current best loss
            best_eval_loss_val = eval_loss_val
            best_epoch = epoch
            
            # save it
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training")

        # early pause if we cannot achieve best loss after 3 epoch
        if epoch - best_epoch  >= 3:
            logger.info("Early stopping, no more improvement")
            sys.exit(0)

if __name__ == "__main__":
    main()