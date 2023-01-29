import argparse
import os
import logging
import numpy as np
import random
from sklearn.svm import LinearSVC

from tqdm.auto import tqdm
from sklearn.linear_model import SGDClassifier

import warnings
warnings.filterwarnings("ignore")

import math

import time

from transformers import (
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import INLP.debias as debias

from utils import FairClassificationMetrics

from dataset_loading import (
    load_raw_pq_data,
    load_text_reps,
    get_Y_labels,
    get_A_labels,
    TensorDataset,
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="data augmentation script")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=7, help="Total number of training epochs to perform.")
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="biasbios",
        choices=["biasbios", "jigsaw-race"],
        help="datasets",
    )
    parser.add_argument(
        "--eval_before_train",
        action="store_true",
        help="evaluation before training",
    )
    # parameters tailer to INLP
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        required=True,
        help="pretrain model name",
    )
    parser.add_argument("--encoded_data_path", type=str, default=None, help="encoded_data_path")
    parser.add_argument("--num_clfs", type=int, default=10, help="num. of clf using in INLP")
    parser.add_argument("--min_acc", type=float, default=0.0, help="above this threshold, ignore the learned classifier")
    parser.add_argument("--A_clf_type", type=str, default="svm", choices=["sgd", "svm"], help="A clf type")
    # A_clf_type
    args = parser.parse_args()
    
    # Sanity checks
    assert args.output_dir is not None
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def get_projection_matrix(num_clfs, X_train, A_train, X_dev, A_dev, Y_train_task, Y_dev_task, dim, args):
    
    start = time.time()

    if args.A_clf_type == "sgd":
        A_clf = SGDClassifier
        params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': 64}
    else:
        A_clf = LinearSVC
        params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}
        
    P,rowspace_projections, Ws = debias.get_debiasing_projection(
        classifier_class=A_clf, 
        cls_params=params, 
        num_classifiers=num_clfs, 
        input_dim=dim, 
        is_autoregressive=True, 
        min_accuracy=args.min_acc,
        X_train=X_train, 
        Y_train=A_train, 
        X_dev=X_dev,
        Y_dev=A_dev,
        Y_train_main=Y_train_task, 
        Y_dev_main=Y_dev_task, 
        by_class = True,
    )
    print("time: {}".format(time.time() - start))
    return P,rowspace_projections, Ws

class MLPModel(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.encoder = nn.Linear(input_size, input_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(
        self, 
        input_ids=None,
        labels=None,
        protected_group_labels=None,
    ):
        hs = self.encoder(input_ids)
        hs = self.activation(hs)
        out = self.classifier(hs)
        return out

def move(batch: dict, device: torch.device) -> dict:
    for (key, val) in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)
        else:
            raise NotImplementedError
    return batch

def eval_model(model, test_dataloader, metrics, args):
    model.eval()
    assert len(metrics) == 0
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            # move batch to proper device
            batch = move(batch, args.device)
            logits = model(**batch)
            predictions = logits.argmax(dim=-1)
            scores = F.softmax(logits, dim=-1)

            # add batch here
            metrics.add_batch(
                predictions=predictions,
                references=batch["labels"],
                scores=scores,
                sensitive_attributes=F.one_hot(batch["protected_group_labels"], num_classes=args.num_protected_group_labels),
            )
        # compute metrics
        eval_metrics, _ = metrics.compute()
    return eval_metrics, _

def main():
    args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
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

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"parser args : {vars(args)}")
    
    # load data (test with biasbios data first)
    raw_dataset, dataset_info = load_raw_pq_data(args)
    raw_train, raw_val, raw_test = raw_dataset['train'], raw_dataset['val'], raw_dataset['test']
    
    # load bert CLS representation
    x_train, x_val, x_test = load_text_reps(args)
    
    assert len(x_train) == len(raw_train)
    assert len(x_val) == len(raw_val)
    assert len(x_test) == len(raw_test)
    
    # get Y labels
    y_train, y_val, y_test = get_Y_labels(raw_train, raw_val, raw_test, dataset_info, args)
    y2i = dataset_info['label_to_id']
    i2y = {v:k for k, v in dataset_info['label_to_id'].items()}
    args.num_labels = len(y2i)
    
    # get A labels
    a_train, a_val, a_test = get_A_labels(raw_train, raw_val, raw_test, dataset_info, args)
    a2i = dataset_info['protected_group_to_id']
    i2a = {v:k for k, v in dataset_info['protected_group_to_id'].items()}
    a_test_label = [i2a[a_idx] for a_idx in a_test]
    args.num_protected_group_labels = len(a2i)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # get metrics
    metrics = FairClassificationMetrics()

    # train INLP and evaluate
    input_dim = x_train.shape[1]
    P, rowspace_projections, Ws = get_projection_matrix(
        num_clfs = args.num_clfs, 
        X_train = x_train, 
        A_train = a_train, 
        X_dev = x_val, 
        A_dev = a_val, 
        Y_train_task = y_train, 
        Y_dev_task = y_val, 
        dim = input_dim,
        args = args,
    )
    
    # get the transformed data and pack it
    transformed_x_train = (P.dot(x_train.T)).T
    transformed_x_val = (P.dot(x_test.T)).T
    transformed_x_test = (P.dot(x_test.T)).T
    
    train_dataset = TensorDataset(X=transformed_x_train, Y=y_train, A=a_train)
    val_dataset = TensorDataset(X=transformed_x_val, Y=y_val, A=a_val)
    test_dataset = TensorDataset(X=transformed_x_test, Y=y_test, A=a_test)
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
    )
    
    # get MLP model and device
    model = MLPModel(input_size=transformed_x_train.shape[1], num_labels=len(y2i))
    model = model.to(args.device)
    
    # Optimizer
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
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
   
    # eval_before_train
    if args.eval_before_train:
        eval_metrics, _ = eval_model(model, test_dataloader, metrics, args)
        logger.info(f"Eval before train: {eval_metrics}")
    
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # move batch to proper device
            batch = move(batch, args.device)
            logits = model(**batch)
            # get CE loss and backward
            labels = batch['labels']
            loss = criterion(logits, labels)
            loss.backward()
            # step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps >= args.max_train_steps:
                break

        eval_metrics, _ = eval_model(model, test_dataloader, metrics, args)
        logger.info(f"epoch {epoch}: {eval_metrics}")



if __name__ == '__main__':
    main()