from typing import Optional, List, Union, Callable, Any, Tuple
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict

import torch
from torch import nn
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.checkpoint import get_device_states, set_device_states
import torch.distributed as dist

from .contrastive_losses import (
    SupConLoss,
    ConSupConLoss
)

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None

class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    
    def __init__(self, model, args, accelerator):
        """
        Initialize the Gradient Cache class instance.
        :param models: encoder models to be updated by the current cache.
        :param args
        :param accelerator
        """
        self.model = model
        self.accelerator = accelerator
        self.model_args = args
        
        self.loss_1_criterion = SupConLoss(temperature=self.model_args.temperature)
        self.loss_2_criterion = ConSupConLoss(temperature=self.model_args.temperature)
        self.aux_loss_weight = self.model_args.aux_loss_weight
        
        self.chunk_size = self.model_args.gradcache_chunk_size
        self._get_input_tensors_strict = False
    
    
    def split_inputs(self, model_input, chunk_size: int):
        r"""
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """

        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, Tensor) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')
    
    def get_input_tensors(self, model_input) -> List[Tensor]:
        r"""
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def forward_no_grad(self, model_inputs):
        r"""
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations [] and b) recorded random states.
            return has the type Tuple[Tensor, List[RandContext]]
        """
        rnd_states = []
        model_reps = []
        labels = []
        protected_group_labels = []

        with torch.no_grad():
            for x in model_inputs:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                z = self.model(**x)
                model_reps.append(z)
                labels.append(x['labels'])
                protected_group_labels.append(x['protected_group_labels'])

        # concatenate all sub-batch representations/label/protected_group_label
        model_reps = torch.cat(model_reps, dim=0)
        labels = torch.cat(labels, dim=0)
        protected_group_labels = torch.cat(protected_group_labels, dim=0)

        return model_reps, rnd_states, labels, protected_group_labels

    def compute_loss(self, reps, **loss_kwargs):
        """
        Compute the loss based on the representation tensors. The tensors should be ordered same as the list of models
        registered in this GradCache class instance.
        :param reps: Representations for computing the loss.
        :param loss_kwargs: Keyword arguments input to the loss function.
        :return: the loss tensor.
        """

        # Separate representation
        z1, z2 = reps[:,0], reps[:,1]
        
        # get batch size, labels and protect_group_labels
        batch_size = reps.size(0)
        labels = loss_kwargs['labels']
        protected_group_labels = loss_kwargs['protected_group_labels']
        
        
        # Gather all embeddings if using distributed training
        # Gather all embeddings if using distributed training
        if dist.is_initialized():

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            label_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            protected_group_label_list = [torch.zeros_like(protected_group_labels) for _ in range(dist.get_world_size())]
            
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            dist.all_gather(tensor_list=label_list, tensor=labels.contiguous())
            dist.all_gather(tensor_list=protected_group_label_list, tensor=protected_group_labels.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
            labels = torch.cat(label_list, 0)
            protected_group_labels = torch.cat(protected_group_label_list, 0)

            # change label and protected_group_labels shape
            # NOTE: only works for single label classification
            labels = labels.squeeze(1)
            protected_group_labels = protected_group_labels.argmax(1)
            assert z1.shape[0] == z2.shape[0] == batch_size*dist.get_world_size()
            assert labels.shape == (batch_size*dist.get_world_size(),)
            assert protected_group_labels.shape == (batch_size*dist.get_world_size(),)

        else:
            # change label and protected_group_labels shape
            # NOTE: only works for single label classification
            labels = labels.squeeze(1)
            protected_group_labels = protected_group_labels.argmax(1)
            assert labels.shape == (batch_size,)
            assert protected_group_labels.shape == (batch_size,)
        
        features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        
        loss_1 = self.loss_1_criterion(
            features=features, 
            labels=labels,
        )
        
        loss_2 = self.loss_2_criterion(
            features=features, 
            labels=labels, 
            protected_group_labels=protected_group_labels,
        )
        
        loss = loss_1 + self.aux_loss_weight * loss_2
        
        loss_dict = {
            "loss": loss,
            "loss_1": loss_1,
            "loss_2": loss_2,
        }
        
        return loss_dict


    def build_cache(self, reps: Tensor, **loss_kwargs):
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """

        reps = reps.detach().requires_grad_()

        with autocast() if self.accelerator.use_fp16 else nullcontext():
            loss_dict = self.compute_loss(reps, **loss_kwargs)
            loss = loss_dict['loss']
        
        self.accelerator.backward(loss)
        
        cache = reps.grad
        
        return cache, {k:v.detach() for k,v in loss_dict.items()}
        
    
    def forward_backward(
            self,
            model_inputs,
            cached_gradients: Tuple[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False
    ):
        r"""
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        
        if no_sync_except_last:
            sync_contexts = [self.model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]
            
        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    z = self.model(**x)
                    # second_reps.append(z)
                surrogate = torch.dot(z.flatten(), gradient.flatten())
                surrogate.backward()
                
        # second_reps = torch.cat(second_reps, dim=0)
        # return second_reps.detach()
    
    def cache_step(self, batch, no_sync_except_last=False):
        
        r"""
        Run a cached step to compute gradient over the inputs.
        :param batch: input batch of examples
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :return: The current's loss.
        """

        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), [self.bert, self.mlp])), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'
        
        # splited_batches: List[Dict]
        # len(splited_batches) == batch_size // chunk_size
        splited_batches = self.split_inputs(batch, self.chunk_size)
        
        # graph-less forward
        # model_reps: Tensor with shape [batch_size, 2, hidden_dim]
        # rnd_states List[RandContext], len(rnd_states) == batch_size // chunk_size
        # labels : Tensor with shape [batch_size, 1]
        # protected_group_labels: Tensor with shape [batch_size, a_dim]
        model_reps, rnd_states, labels, protected_group_labels = self.forward_no_grad(splited_batches)


        # build cache
        # cache has the type Tuple[Tensor]
        # each tensor in cache has the shape [chunk_size, 2, hidden_dim]
        loss_kwargs = {
            "labels": labels, 
            "protected_group_labels": protected_group_labels,
        }
        cache, loss_dict = self.build_cache(model_reps, **loss_kwargs)
        cache = cache.split(self.chunk_size, dim=0)
        
        self.forward_backward(
            splited_batches, 
            cache, 
            rnd_states, 
            no_sync_except_last=no_sync_except_last,
        )
        
        return loss_dict

    def forward(self, model_inputs):
        r"""
        forward pass w/ or w/o gradient
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations [] and b) recorded random states.
            return has the type Tuple[Tensor, List[RandContext]]
        """
        rnd_states = []
        model_reps = []
        labels = []
        protected_group_labels = []

        for x in model_inputs:
            rnd_states.append(RandContext(*self.get_input_tensors(x)))
            z = self.model(**x)
            model_reps.append(z)
            labels.append(x['labels'])
            protected_group_labels.append(x['protected_group_labels'])

        # concatenate all sub-batch representations/label/protected_group_label
        model_reps = torch.cat(model_reps, dim=0)
        labels = torch.cat(labels, dim=0)
        protected_group_labels = torch.cat(protected_group_labels, dim=0)

        return model_reps, rnd_states, labels, protected_group_labels     
        
    def step(self, batch, is_backward=False):

        r"""
        Run a forwardd step to compute loss over the inputs.
        :param batch: input batch of examples
        :return: The current's loss.
        """

        splited_batches = self.split_inputs(batch, self.chunk_size)

        model_reps, rnd_states, labels, protected_group_labels = self.forward(splited_batches)


        loss_kwargs = {
            "labels": labels, 
            "protected_group_labels": protected_group_labels,
        }

        with autocast() if self.accelerator.use_fp16 else nullcontext():
            loss_dict = self.compute_loss(model_reps, **loss_kwargs)

        if is_backward:
            loss = loss_dict['loss']
            self.accelerator.backward(loss)
            loss_dict = {k:v.detach() for k,v in loss_dict.items()}

        return loss_dict