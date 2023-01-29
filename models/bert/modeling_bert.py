#!/usr/bin/env python
# coding=utf-8

"""Code adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""
from dataclasses import dataclass
from typing import Optional, List, Union, Callable, Any, Tuple


from transformers import (
    BertPreTrainedModel,
    BertModel,
)
from transformers.file_utils import ModelOutput 
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.distributed as dist

from ..contrastive_losses import (
    SupConLoss,
    ConSupConLoss
)

class BertForSequenceClassification(BertPreTrainedModel):
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

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
        
@dataclass
class CLModelOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`,  *optional*
        loss_1 (`torch.FloatTensor` of shape `(1,)`,  *optional*
        loss_2 (`torch.FloatTensor` of shape `(1,)`,  *optional*
    """

    loss: Optional[torch.FloatTensor] = None
    loss_1: Optional[torch.FloatTensor] = None
    loss_2: Optional[torch.FloatTensor] = None
    
        
class BertForContrastiveLearning(BertPreTrainedModel):
    def __init__(self, config,  *model_args, **model_kargs):
        super().__init__(config)
        self.config = config
        self.model_args = model_kargs["model_args"]
        
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # other module for contrastive learning (pooler type, loss function, etc.)
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)

        if self.model_args.pooler_type == "cls":
            self.mlp = MLPLayer(config)
        
        self.loss_1_criterion = SupConLoss(temperature=self.model_args.temperature)
        self.loss_2_criterion = ConSupConLoss(temperature=self.model_args.temperature)
        self.aux_loss_weight = self.model_args.aux_loss_weight
         
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
        Bert for fair CL pretraining
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance
        num_sent = input_ids.size(1)
        
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        if attention_mask is not None:
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        # Get raw embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        
        # Pooling
        pooler_output = self.pooler(attention_mask, outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)
        
        # Separate representation
        z1, z2 = pooler_output[:,0], pooler_output[:,1]
        
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
                
        if not return_dict:
            return (loss, loss_1, loss_2)

        return CLModelOutput(
            loss=loss,
            loss_1=loss_1,
            loss_2=loss_2,
        )

class BertForCLGradCache(BertPreTrainedModel):
    def __init__(self, config,  *model_args, **model_kargs):
        super().__init__(config)
        self.config = config
        self.model_args = model_kargs["model_args"]
        
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # other module for contrastive learning (pooler type, loss function, etc.)
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)

        if self.model_args.pooler_type == "cls":
            self.mlp = MLPLayer(config)

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
        Bert for fair CL pretraining with grad cache
        Just forward to get reps
        """

        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance
        num_sent = input_ids.size(1)
        
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        if attention_mask is not None:
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        # Get raw embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        
        # Pooling
        pooler_output = self.pooler(attention_mask, outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)
        
        return pooler_output

class BertForOneStageCLGradCache(BertPreTrainedModel):
    def __init__(self, config,  *model_args, **model_kargs):
        super().__init__(config)
        self.config = config
        self.model_args = model_kargs["model_args"]
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # other module for contrastive learning (pooler type, loss function, etc.)
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)

        if self.model_args.pooler_type == "cls":
            self.mlp = MLPLayer(config)

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
        Bert for fair one_stage CL with grad cache
        Just forward to get reps and logits
        """

        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance
        num_sent = input_ids.size(1)
        
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        if attention_mask is not None:
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        # Get raw embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        
        # Pooling
        pooler_output = self.pooler(attention_mask, outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)
        
        # get logits for classifcation
        logits = self.classifier(pooler_output)
        
        return pooler_output, logits