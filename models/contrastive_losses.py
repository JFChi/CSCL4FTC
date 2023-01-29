import torch
import torch.nn as nn
from torch.nn import functional as F

import logging

logger = logging.getLogger(__name__)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Codes modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py"""
    def __init__(self, temperature=0.1, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # [bsz, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # shape [bsz, bsz]
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # contrast_count=2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape [bsz*2, feature_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # shape [bsz, feature_dim]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # shape [bsz*2, feature_dim]
            anchor_count = contrast_count # anchor_count=2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # compute logits: each element (logits_{i, j}) in logit correspond to (feat_i * feat_j) / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'

        # tile mask: in supcon loss, the mask is label, so need to repeat vertically and horizontally
        mask = mask.repeat(anchor_count, contrast_count) 
        # mask-out self-contrast cases (only the diagonal element is zero, otherwise 1)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), # shape [bsz*2, 1] if contrast_mode is 'all'; shape [bsz, 1] if contrast_mode is 'one'
            0
        ) # shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'
        mask = mask * logits_mask # shape same as logits_mask
        
        # for numerical stability and self-mask-out
        anchor_dot_contrast = anchor_dot_contrast * logits_mask
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # shape [bsz*2, 1] if all; shape [bsz, 1] if one 
        logits = anchor_dot_contrast - logits_max.detach() # shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'
        
        # compute log_prob: shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive: shape (anchor_count * batch_size)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        # assert loss is not nan or inf
        assert not torch.any(torch.isnan(loss))
        assert not torch.any(torch.isinf(loss))

        return loss
    
class SimpleContrastiveLoss:
    """
    Code adapted from https://github.com/luyug/GradCache/blob/main/src/grad_cache/loss.py"""
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 0.1):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature

    def __call__(self, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor = None, reduction: str = 'mean'):
        if target is None:
            assert x.size(0) * self.target_per_qry == y.size(0)
            target = torch.arange(0, x.size(0) * self.target_per_qry, self.target_per_qry, device=x.device)

        logits = torch.matmul(x, y.transpose(0, 1)) / self.temperature
        
        simcont_loss = self.temperature * F.cross_entropy(logits, target, reduction=reduction)
        
        # assert loss is not nan or inf
        assert not torch.any(torch.isnan(simcont_loss))
        assert not torch.any(torch.isinf(simcont_loss))
        
        return simcont_loss


class SimpleContrastiveLossV2:
    """
    Code adapted from https://github.com/luyug/GradCache/blob/main/src/grad_cache/loss.py"""
    def __init__(self, n_hard_negatives: int = 0, temperature: float = 0.1):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature

    def __call__(self, x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean'):

        ### TODO check correctness of simclr in supconloss ###
        features = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        contrast_count = features.shape[1] # contrast_count=2
        contrastive_mode = 'all'

        batch_size = features.shape[0]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrastive_mode == 'one':
            anchor_feature = features[:, 0] # shape [bsz, feature_dim]
            anchor_count = 1
        elif contrastive_mode == 'all':
            anchor_feature = contrast_feature # shape [bsz*2, feature_dim]
            anchor_count = contrast_count # anchor_count=2
        else:
            raise NotImplementedError
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_mask = torch.scatter(
            torch.ones_like(anchor_dot_contrast),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1), # shape [bsz*2, 1] if contrast_mode is 'all'; shape [bsz, 1] if contrast_mode is 'one'
            0
        ) 

        logits_v2 = anchor_dot_contrast * logits_mask
        target_v2_1 = torch.arange(x.size(0), 2*x.size(0), 1, device=x.device)
        target_v2_2 = torch.arange(0, x.size(0), 1, device=x.device)
        target_v2 = torch.cat((target_v2_1, target_v2_2), dim=0)
        ssl_loss = self.temperature * F.cross_entropy(logits_v2, target_v2, reduction=reduction)

        assert not torch.any(torch.isnan(ssl_loss))
        assert not torch.any(torch.isinf(ssl_loss))

        return ssl_loss

class ConSupConLoss(nn.Module):
    """ Conditional Supervised Contrastive Loss
    Codes modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py"""
    def __init__(self, temperature=0.1, contrast_mode='all'):
        super(ConSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature

    def forward(self, features, labels=None, protected_group_labels=None):
        """Compute loss for model, 
        where the positive examples are different views of the same examples,
        and negative examples are the examples share the same Y and A labels

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            protected_group_labels: protected group label of shape [bsz].
            
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        # mask: contrastive mask for positive examples shape [bsz, bsz]
        # torch.eye(batch_size) as the initial input value
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        
        if (labels is None) or (protected_group_labels is None):
            raise ValueError('`labels` and `mask` must be specified')
        
        # create label mask
        labels = labels.contiguous().view(-1, 1) # [bsz, 1]
        label_mask = torch.eq(labels, labels.T).float().to(device) # shape [bsz, bsz]

        # create protected group label mask
        protected_group_labels = protected_group_labels.contiguous().view(-1, 1) # [bsz, 1]
        protected_group_label_mask = torch.eq(protected_group_labels,
            protected_group_labels.T).float().to(device)

        # mask that share the same Y and A values (self-included)
        negative_mask = label_mask * protected_group_label_mask # * (1 - mask)

        contrast_count = features.shape[1] # contrast_count=2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape [bsz*2, feature_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # shape [bsz, feature_dim]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # shape [bsz*2, feature_dim]
            anchor_count = contrast_count # anchor_count=2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # compute logits: each element (logits_{i, j}) in logit correspond to (feat_i * feat_j) / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'

        # tile mask: in supcon loss, the mask is label, so need to repeat vertically and horizontally
        mask = mask.repeat(anchor_count, contrast_count) 
        negative_mask = negative_mask.repeat(anchor_count, contrast_count) 
        # mask-out self-contrast cases (only the diagonal element is zero, otherwise 1)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), # shape [bsz*2, 1] if contrast_mode is 'all'; shape [bsz, 1] if contrast_mode is 'one'
            0
        ) # shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'
        mask = mask * logits_mask # shape same as logits_mask
        
        # for numerical stability, self-mask-out, and mask out the irrelevant examples 
        anchor_dot_contrast = anchor_dot_contrast * logits_mask * negative_mask
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # shape [bsz*2, 1] if all; shape [bsz, 1] if one 
        logits = anchor_dot_contrast - logits_max.detach() # shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'
        
        # compute log_prob: shape [bsz*2, bsz*2] if contrast_mode is 'all'; shape [bsz, bsz*2] if contrast_mode is 'one'
        # exp_logits is the denominator
        exp_logits = torch.exp(logits) * logits_mask * negative_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive: shape (anchor_count * batch_size)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        assert not torch.any(torch.isnan(loss))
        assert not torch.any(torch.isinf(loss))
        
        return loss
