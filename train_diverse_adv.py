#!/usr/bin/env python
# coding=utf-8

"""fair adverarial training with diverse adversaries"""

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import (
    Union,
    Any,
    Dict,
)

import torch
import torch.nn.functional as F
from torch.optim import Adam

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
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from utils import FairClassificationMetrics

from adversarial_training.models import (
    DiffLoss,
    Discriminator,
    BertForAdversarialTraining,
)

from adversarial_training.utils import (
    train_epoch,
    eval_epoch,
    adv_train_eval,
)

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
        "--logging_steps", type=int, default=500, help="Number of steps for logging the train loss."
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
        help="if save model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jigsaw-race",
        choices=["biasbios", "jigsaw-race"],
        help="datasets",
    )
    # argument for divserse adversarial training
    parser.add_argument(
        "--n_discriminators",
        type=int,
        default=3,
        help="Number of discriminators to train",
    )
    parser.add_argument(
        "--adv_hidden_units",
        type=int,
        default=256,
        help="hidden unit of discriminators",
    )
    parser.add_argument(
        "--adv_train_batch_size",
        type=int,
        default=256,
        help="train_batch_size for ",
    )
    parser.add_argument(
        "--adv_training_epochs",
        type=int,
        default=10,
        help=(
            "Number of epochs that should be used to train the adversaries "
            "within each training epoch"
        ),
    )
    parser.add_argument(
        "--lambda_adv",
        type=float,
        default=1.0,
        help=(
            "Tunes the tradeoff between predictions vs adversary "
            "performance in model training"
        ),
    )
    parser.add_argument(
        "--lambda_diff",
        type=float,
        default=5000,
        help=(
            "Tunes the tradeoff between adversary performance "
            "and orthogonality in adversary training"
        ),
    )
    parser.add_argument(
        "--by_class",
        action="store_true",
        help="whether add label information into adversarial training",
    )
    args = parser.parse_args()
    
    # Sanity checks
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        assert args.model_name_or_path in ['bert-base-uncased']
    return args

def main():
    args = parse_args()
    
    # in the baseline script, 
    # we avoid parallel training using accelerator
    # and use single device + gradient accumulation step for training 
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        # tmp path used for saving discriminators
        os.makedirs(os.path.join(args.output_dir, "tmp"), exist_ok=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, f"train_{args.seed}.log")),
            logging.StreamHandler(),
        ],
    )
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    
    # load datasets and tokenizer, and preprocess datasets
    if args.dataset == 'biasbios':
        from dataset_loading import load_biasbios_for_ce
        
        # load tokenizer and preprocessing the datasets    
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=not args.use_slow_tokenizer
        )
        processed_dataset, dataset_info  = load_biasbios_for_ce(tokenizer, args, accelerator=None)
        
        id_to_label = dataset_info["id_to_label"]
        label_to_id = dataset_info["label_to_id"]
        num_labels = dataset_info["num_labels"]
        
        train_dataset = processed_dataset["train"]
        val_dataset = processed_dataset["val"]
        test_dataset = processed_dataset["test"]
        
    elif args.dataset == 'jigsaw-race':
        from dataset_loading import load_jigsaw_race_for_ce
        
        # load tokenizer and preprocessing the datasets    
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=not args.use_slow_tokenizer
        )
        
        processed_dataset, dataset_info  = load_jigsaw_race_for_ce(tokenizer, args, accelerator=None)
        
        id_to_label = dataset_info["id_to_label"]
        label_to_id = dataset_info["label_to_id"]
        num_labels = dataset_info["num_labels"]
        
        train_dataset = processed_dataset["train"]
        val_dataset = processed_dataset["val"]
        test_dataset = processed_dataset["test"]

    else:
        raise NotImplementedError
    
    # Load pretrained model and config based on datasets
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, 
        num_labels=num_labels
    )
    if args.model_name_or_path == 'bert-base-uncased':
        model = BertForAdversarialTraining.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        raise NotImplementedError
    model.config.label2id = label_to_id
    model.config.id2label = id_to_label
    # put model to device
    model = model.to(args.device)

    # Log some parameters here!
    logger.info(f"parser args : {vars(args)}")

    # load discriminators
    num_protected_labels = len(dataset_info["protected_group_to_id"])
    num_labels = len(dataset_info["label_to_id"])
    adv_input_size = config.hidden_size # NOTE BERT-base hidden size
    discriminators = [Discriminator(args, input_size=adv_input_size, num_classes=num_protected_labels, num_labels=num_labels) for _ in range(args.n_discriminators)]
    discriminators = [dis.to(args.device) for dis in discriminators]

    diff_loss = DiffLoss()
    args.diff_loss = diff_loss
    
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset, 
        collate_fn=data_collator, 
        batch_size=args.adv_train_batch_size, # used for evaluation the discriminator
    )
    test_dataloader = DataLoader(
        test_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
    )
    # train loader for adv discriminators
    adv_train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.adv_train_batch_size,
    )

    # Optimizer and adv optimizers
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
    adv_optimizers = [Adam(filter(lambda p: p.requires_grad, dis.parameters()), lr=args.learning_rate) for dis in discriminators]


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
    
    # load metric
    metrics = FairClassificationMetrics()

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total train batch size for adversarial discriminators = {args.adv_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if args.eval_before_train:
        eval_metrics, _, adv_loss_test = eval_epoch(
            model=model,
            discriminators=discriminators,
            iterator = test_dataloader, 
            metrics=metrics, 
            device =args.device, 
            args=args
        )
        logger.info(f"Eval before train: {eval_metrics}")
        logger.info(f"adv loss for test set: {adv_loss_test:.6f}")


    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.num_train_epochs):
        
        # train and eval adv discriminators
        adv_train_eval(
            model=model, 
            discriminators=discriminators, 
            train_iterator=adv_train_dataloader,
            valid_iterator=eval_dataloader,
            adv_optimizers=adv_optimizers, 
            criterion=criterion, 
            device=args.device, 
            args=args,
        )

        # train main components 
        completed_steps = train_epoch(
            model=model, 
            discriminators=discriminators, 
            iterator=train_dataloader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=args.device, 
            args=args,
            lr_scheduler=lr_scheduler,
            progress_bar=progress_bar,
            completed_steps=completed_steps,
        )

        # evaluate model and discriminator using test set
        eval_metrics, _, adv_loss_test = eval_epoch(
            model=model, 
            discriminators=discriminators,
            iterator=test_dataloader, 
            metrics=metrics, 
            device =args.device, 
            args=args,
        )

        logger.info(f"epoch {epoch}: {eval_metrics}")
        logger.info(f"adv loss for test set: {adv_loss_test:.6f}")


if __name__ == "__main__":
    main()