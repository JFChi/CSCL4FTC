#!/usr/bin/env python
# coding=utf-8

# code adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py 
""" Finetuning model for sequence classification with no trainer."""

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

from models.bert.modeling_bert import BertForSequenceClassification

from utils import FairClassificationMetrics

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
    args = parser.parse_args()

    # Sanity checks
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        assert args.model_name_or_path in ['bert-base-uncased']
    return args

def eval_model(model, test_dataloader, metrics, accelerator):
    model.eval()
    assert len(metrics) == 0
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            scores = F.softmax(outputs.logits, dim=-1)
            # add batch here
            metrics.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
                scores=accelerator.gather(scores),
                sensitive_attributes=accelerator.gather(batch["protected_group_labels"]),
            )
        
        # compute metrics
        eval_metrics, _ = metrics.compute()
    return eval_metrics, _

def compute_validation_loss(model, eval_dataloader, accelerator):
    model.eval()
    with torch.no_grad():
        completed_eval_steps = 0
        eval_loss_val = torch.tensor(0.0).to(accelerator.device)
        for _, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            eval_loss_val += accelerator.gather(outputs.loss).mean()
            completed_eval_steps += 1
    
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
        from dataset_loading import load_biasbios_for_ce
        
        # load tokenizer and preprocessing the datasets    
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=not args.use_slow_tokenizer
        )
        processed_dataset, dataset_info  = load_biasbios_for_ce(tokenizer, args, accelerator)
        
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
        
        processed_dataset, dataset_info  = load_jigsaw_race_for_ce(tokenizer, args, accelerator)
        
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
    if args.model_name_or_path == 'bert-base-uncased':
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        raise NotImplementedError
    model.config.label2id = label_to_id
    model.config.id2label = id_to_label
    
    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 2):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
    # Log some parameters here!
    logger.info(f"parser args : {vars(args)}")
    
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

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

    # load metric
    metrics = FairClassificationMetrics()

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    if args.eval_before_train:
        eval_metrics, _ = eval_model(model, test_dataloader, metrics, accelerator)
        logger.info(f"Eval before train: {eval_metrics}")

    best_eval_loss_val = torch.tensor(float('inf')).to(accelerator.device)
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps >= args.max_train_steps:
                break
        
        eval_metrics, _ = eval_model(model, test_dataloader, metrics, accelerator)
        logger.info(f"epoch {epoch}: {eval_metrics}")

        if epoch < args.num_train_epochs - 1 and args.save_model:
            eval_loss_val = compute_validation_loss(model, eval_dataloader, accelerator)
            if eval_loss_val.item() <= best_eval_loss_val.item():
                # reset current best loss
                best_eval_loss_val = eval_loss_val
                logger.info(f"achieve best val loss at epoch {epoch}: {best_eval_loss_val.item()}")

                # save the model with the smallest validation loss
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                model_path_best = os.path.join(args.output_dir, 'best')
                unwrapped_model.save_pretrained(model_path_best, save_function=accelerator.save)
                tokenizer.save_pretrained(model_path_best)
                if args.push_to_hub and accelerator.is_main_process:
                    repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)


    if args.output_dir is not None and args.save_model:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        model_path_last = os.path.join(args.output_dir, 'last')
        unwrapped_model.save_pretrained(model_path_last, save_function=accelerator.save)
        logger.info(f"Save model after training ...")
        if accelerator.is_main_process:
            tokenizer.save_pretrained(model_path_last)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training")


if __name__ == "__main__":
    main()