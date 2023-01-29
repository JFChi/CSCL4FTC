import logging
import os
from typing import (
    Dict,
)

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

def move(batch: dict, device: torch.device) -> dict:
    for (key, val) in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)
        else:
            raise NotImplementedError
    return batch

def train_epoch(model, discriminators, iterator, optimizer, criterion, device, args, **kwargs):

    # get **kwargs
    lr_scheduler = kwargs.pop('lr_scheduler')
    progress_bar = kwargs.pop('progress_bar')
    completed_steps = kwargs.pop('completed_steps')

    # activate train mode
    model.train()
    
    for discriminator in discriminators:
        discriminator.train()
        
    # activate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = True
    
    running_main_loss_val, running_adv_loss_val = 0.0, 0.0
    running_total_loss_val = 0.0
    for step, batch in enumerate(iterator):
        # move batch to proper device
        batch = move(batch, device)

        # forward
        logits, hs = model(**batch)

        # get CE loss for main task
        labels = batch['labels']
        main_loss = criterion(logits, labels)

        # get adversarial losses
        adv_loss = torch.tensor(0.0).to(device)
        protected_group_labels = torch.argmax(batch['protected_group_labels'], dim=-1)
        for discriminator in discriminators:
            if args.by_class:
                adv_predictions, _ = discriminator(hs, labels=labels)
            else:
                adv_predictions, _ = discriminator(hs)
            adv_loss += criterion(adv_predictions, protected_group_labels) / len(discriminators)
        
        # compute total loss and gradient accumulation 
        total_loss = main_loss + adv_loss
        total_loss = total_loss / args.gradient_accumulation_steps
        total_loss.backward()

        if step % args.gradient_accumulation_steps == 0 or step == len(iterator) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1
            
        if completed_steps >= args.max_train_steps:
            break
    
        # accumulate value for logging
        running_main_loss_val += main_loss.item()
        running_adv_loss_val += adv_loss.item()
        running_total_loss_val += total_loss.item()

        # logging every N steps
        if completed_steps % args.logging_steps == 0:

            log_info: Dict[str, float] = {}
            log_info['main_loss'] = round(running_main_loss_val / step, 6)
            log_info['adv_loss'] = round(running_adv_loss_val / step, 6)
            log_info['total_loss'] = round(running_total_loss_val / step, 6)
        
            # log it
            logger.info(f"logging training loss: {log_info}")

    return completed_steps

def eval_epoch(model, discriminators, iterator, metrics, device, args):

    # activate test mode
    model.eval()
    for discriminator in discriminators:
        discriminator.eval()
    
    assert len(metrics) == 0

    running_adv_loss_val = 0.0
    adv_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            # move batch to proper device
            batch = move(batch, device)

            # forward
            logits, hs = model(**batch)
            predictions = logits.argmax(dim=-1)
            scores = F.softmax(logits, dim=-1)

            # add batch here
            metrics.add_batch(
                predictions=predictions,
                references=batch["labels"],
                scores=scores,
                sensitive_attributes=batch["protected_group_labels"],
            )

            # get and log advserserial losses
            protected_group_labels = torch.argmax(batch['protected_group_labels'], dim=-1)
            labels = batch['labels']
            for discriminator in discriminators: 
                if args.by_class:
                    adv_predictions, _ = discriminator(hs, labels=labels)
                else:
                    adv_predictions, _ = discriminator(hs)
                running_adv_loss_val += adv_criterion(adv_predictions, protected_group_labels).item() / len(discriminators)
        
        running_adv_loss_val = running_adv_loss_val / len(iterator)

        # compute metrics
        eval_metrics, _ = metrics.compute()
        return eval_metrics, _, running_adv_loss_val


def adv_train_eval(model, discriminators, train_iterator, valid_iterator, adv_optimizers, criterion, device, args):
    
    best_adv_epoch = -1
    best_eval_loss_val = adv_eval_epoch(
            model=model, 
            discriminators=discriminators, 
            iterator=valid_iterator, 
            criterion=criterion, 
            device=device, 
            args=args,
        )
    logger.info(f"Before training discriminator, adv valid loss {best_eval_loss_val:.6f}")
    for k in range(args.adv_training_epochs):
        train_info = adv_train_epoch(
            adv_epoch=k,
            model=model, 
            discriminators=discriminators, 
            iterator=train_iterator, 
            adv_optimizers=adv_optimizers, 
            criterion=criterion, 
            device=device, 
            args=args,
        )

        logger.info(f"In adv epoch {k}, training info: {train_info}")
        eval_loss_val = adv_eval_epoch(
            model=model, 
            discriminators=discriminators, 
            iterator=valid_iterator, 
            criterion=criterion, 
            device=device, 
            args=args,
        )
        
        if eval_loss_val <= best_eval_loss_val:
            best_eval_loss_val = eval_loss_val
            best_adv_epoch = k
            logger.info(f"In adv epoch {best_adv_epoch}, achieve best adv valid loss {best_eval_loss_val:.6f}")
            # save discriminators
            for j in range(len(discriminators)):
                torch.save(discriminators[j].state_dict(), os.path.join(args.output_dir, "tmp", f"discriminator_{j}"))
        else:
            if best_adv_epoch + 3 <= k:
                logger.info(f"early stopping at adv epoch {k}")
                break

    logger.info(f"Loading the best discriminators after training")
    for j in range(len(discriminators)):
        discriminators[j].load_state_dict(torch.load(os.path.join(args.output_dir, "tmp", f"discriminator_{j}")))

    

def adv_train_epoch(adv_epoch, model, discriminators, iterator, adv_optimizers, criterion, device, args):

    model.eval()
    for discriminator in discriminators:
        discriminator.train()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False


    total_loss_val, adv_loss_val, diff_loss_val = 0.0, 0.0, 0.0
    for batch in tqdm(iterator, desc=f"training discriminators, adv_epoch {adv_epoch}"):
        # move batch to proper device
        batch = move(batch, device)

        with torch.no_grad():
            # graph-less forward
            _, hs = model(**batch)

        # train discriminators using protected labels
        protected_group_labels = torch.argmax(batch['protected_group_labels'], dim=-1)
        labels = batch['labels']
        # iterate all discriminators and train it
        for i, (discriminator, adv_optimizer) in enumerate(zip(discriminators, adv_optimizers)):
            adv_optimizer.zero_grad()
            if args.by_class:
                adv_predictions, adv_hs = discriminator(hs, labels=labels)
            else:
                adv_predictions, adv_hs = discriminator(hs)

            adv_loss = criterion(adv_predictions, protected_group_labels)
        
            # encrouge orthogonality between adv_hs for different discriminators
            difference_loss = torch.tensor(0.0).to(device)
            for j, discriminator_j in enumerate(discriminators):
                if i != j:
                    if args.by_class:
                        _, adv_hs_j = discriminator_j(hs, labels=labels)
                    else:
                        _, adv_hs_j = discriminator_j(hs)
                    # calculate diff_loss (exclude the current model)
                    difference_loss += args.lambda_diff * args.diff_loss(adv_hs, adv_hs_j)
            
            total_loss = adv_loss + difference_loss
            total_loss.backward()
            adv_optimizer.step()

            # logging loss
            total_loss_val += total_loss.item()
            adv_loss_val += adv_loss.item()
            diff_loss_val += difference_loss.item()

    loss_info = {
        'total_loss': total_loss_val / ( len(iterator) * len(discriminators) ),
        'adv_loss': adv_loss_val / ( len(iterator) * len(discriminators) ),
        'diff_loss': diff_loss_val / ( len(iterator) * len(discriminators) ),
    }

    return loss_info



def adv_eval_epoch(model, discriminators, iterator, criterion, device, args):

    model.eval()
    for discriminator in discriminators:
        discriminator.eval()
    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    
    eval_loss_val = 0.0
    with torch.no_grad():
        for batch in iterator:
            # move batch to proper device
            batch = move(batch, device)
            _, hs = model(**batch)
            
            # get protected labels
            protected_group_labels = torch.argmax(batch['protected_group_labels'], dim=-1)
            labels = batch['labels']
            for i, discriminator in enumerate(discriminators):
                if args.by_class:
                    adv_predictions, _ = discriminator(hs, labels=labels)
                else:
                    adv_predictions, _ = discriminator(hs)
                eval_loss = criterion(adv_predictions, protected_group_labels)
                eval_loss_val += eval_loss.item()

    eval_loss_val = eval_loss_val / ( len(iterator) * len(discriminators) )
    return eval_loss_val

    