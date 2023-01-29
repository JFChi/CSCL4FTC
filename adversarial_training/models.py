
from dataclasses import dataclass
from operator import concat
from typing import Optional, List, Union, Callable, Any, Tuple


from transformers import (
    BertPreTrainedModel,
    BertModel,
)
from transformers.file_utils import ModelOutput 
from transformers.modeling_outputs import SequenceClassifierOutput


import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class DiffLoss(torch.nn.Module):
    '''
    From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    '''

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)




class Discriminator(nn.Module):
    def __init__(self, args, input_size, num_classes, num_labels):
        super(Discriminator, self).__init__()

        self.GR = False
        self.grad_rev = GradientReversal(args.lambda_adv)
        if args.by_class:
            input_size = input_size + num_labels
        self.fc1 = nn.Linear(input_size, args.adv_hidden_units)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(args.adv_hidden_units, args.adv_hidden_units)
        self.fc3 = nn.Linear(args.adv_hidden_units, num_classes)

        self.num_labels = num_labels
        self.by_class = args.by_class

    def forward(self, input, labels=None):
        if self.by_class:
            assert labels is not None
            one_hot_labels = F.one_hot(labels, num_classes=self.num_labels)
            input = torch.cat((input, one_hot_labels), dim=-1)

        if self.GR:
            input = self.grad_rev(input)
            
        out = self.fc1(input)
        out = self.LeakyReLU(out)
        adv_hs = self.fc2(out)
        out = self.fc3(adv_hs)
        return out, adv_hs

class BertForAdversarialTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # NOTE: init_weights is wrapped into post_init in newer version of huggingface transformer
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        protected_group_labels = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # return the hidden output of the first token before pooling
        last_hidden_state = outputs[0]
        first_token_tensor = last_hidden_state[:, 0]

        return logits, first_token_tensor

class MLPForAdversarialTraining(nn.Module):
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
        logits = self.classifier(hs)
        return logits, hs