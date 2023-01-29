#!/usr/bin/env python
# coding=utf-8

import random
import os
import re
from collections import Counter
from typing import Optional
import numpy as np

import sklearn.metrics as skm
from sklearn.metrics import (
    accuracy_score, 
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)

import pyarrow as pa

import datasets
from datasets.utils import temp_seed
from datasets.utils.logging import get_logger

logger = get_logger(__name__)


def set_cl_eval_mode(model):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def rms_diff(tpr_diff):
    return np.sqrt(np.mean(tpr_diff**2))

def _get_unique_labels(labels):
    unique_labels = list(np.unique(labels))
    return unique_labels

def div0( a, b, fill=0.0):
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
    if np.isscalar( c ):
        return c if np.isfinite( c ) \
            else fill
    else:
        c[ ~ np.isfinite( c )] = fill
        return c

def compute_deo_for_multi_label_loop(predictions, references, sensitive_attributes):
    '''compute deo based on loops
    where sensitive_attributes is (multilabel) one-hot encoding
    '''
    assert len(references) != 0
    assert len(predictions) == len(references) == len(sensitive_attributes)
    
    unique_y_values = _get_unique_labels(np.vstack((predictions, references)))
    num_a_values = sensitive_attributes.shape[1]

    overall_fpr_dict, group_fpr_dict = {}, {}
    overall_fnr_dict, group_fnr_dict = {}, {}
    overall_tpr_dict, group_tpr_dict = {}, {}
    overall_tnr_dict, group_tnr_dict = {}, {}
    
    # use loop to compute
    for y in unique_y_values:
        
        pos_label_ids = references == y
        neg_label_ids = references != y
        
        fp_ids = (references != y) & (predictions == y)
        fn_ids = (references == y) & (predictions != y)
        tp_ids = (references == y) & (predictions == y)
        tn_ids = (references != y) & (predictions != y)
        
        fpr = np.count_nonzero(fp_ids) / np.count_nonzero(neg_label_ids) if np.count_nonzero(neg_label_ids) != 0 else 0
        fnr = np.count_nonzero(fn_ids) / np.count_nonzero(pos_label_ids) if np.count_nonzero(pos_label_ids) != 0 else 0
        tpr = np.count_nonzero(tp_ids) / np.count_nonzero(pos_label_ids) if np.count_nonzero(pos_label_ids) != 0 else 0
        tnr = np.count_nonzero(tn_ids) / np.count_nonzero(neg_label_ids) if np.count_nonzero(neg_label_ids) != 0 else 0
        
        acc = np.count_nonzero(tp_ids | tn_ids) / len(references)
        
        overall_fpr_dict[y] = fpr
        overall_fnr_dict[y] = fnr
        overall_tpr_dict[y] = tpr
        overall_tnr_dict[y] = tnr
        
        for a_idx in range(num_a_values):
            
            group_pos_label_ids = (references == y) & (sensitive_attributes[:, a_idx] == 1)
            group_neg_label_ids = (references != y) & (sensitive_attributes[:, a_idx] == 1)
        
            group_fp_ids = (references != y) & (predictions == y) & (sensitive_attributes[:, a_idx] == 1)
            group_fn_ids = (references == y) & (predictions != y) & (sensitive_attributes[:, a_idx] == 1)
            group_tp_ids = (references == y) & (predictions == y) & (sensitive_attributes[:, a_idx] == 1)
            group_tn_ids = (references != y) & (predictions != y) & (sensitive_attributes[:, a_idx] == 1)
        
            group_fpr = np.count_nonzero(group_fp_ids) / np.count_nonzero(group_neg_label_ids) if np.count_nonzero(group_neg_label_ids) != 0 else 0
            group_fnr = np.count_nonzero(group_fn_ids) / np.count_nonzero(group_pos_label_ids) if np.count_nonzero(group_pos_label_ids) != 0 else 0
            group_tpr = np.count_nonzero(group_tp_ids) / np.count_nonzero(group_pos_label_ids) if np.count_nonzero(group_pos_label_ids) != 0 else 0
            group_tnr = np.count_nonzero(group_tn_ids) / np.count_nonzero(group_neg_label_ids) if np.count_nonzero(group_neg_label_ids) != 0 else 0
            
            # group_acc = np.count_nonzero(group_tp_ids | group_tn_ids) / np.count_nonzero(sensitive_attributes[:, a_idx] == 1)
        
            group_fpr_dict[(y, a_idx)] = group_fpr
            group_fnr_dict[(y, a_idx)] = group_fnr
            group_tpr_dict[(y, a_idx)] = group_tpr
            group_tnr_dict[(y, a_idx)] = group_tnr
    
    overall_metric_rates = {
        'overall_fpr_dict': overall_fpr_dict,
        'overall_fnr_dict': overall_fnr_dict,
        'overall_tpr_dict': overall_tpr_dict,
        'overall_tnr_dict': overall_tnr_dict,
    }
    group_metric_rates = {
        'group_fpr_dict': group_fpr_dict,
        'group_fnr_dict': group_fnr_dict,
        'group_tpr_dict': group_tpr_dict,
        'group_tnr_dict': group_tnr_dict,
    } 
    
    return group_metric_rates, overall_metric_rates

def compute_deo_for_multi_label_cnf_matrix(predictions, references, sensitive_attributes):
    '''compute deo based on confusion matrix 
    where sensitive_attributes is (multilabel) one-hot encoding
    '''
    unique_y_values = _get_unique_labels(np.vstack((predictions, references)))
    num_a_values = sensitive_attributes.shape[1]
    
    # use matrix to compute
    cnf_matrix = skm.confusion_matrix(
        y_true=references, 
        y_pred=predictions, 
        labels=unique_y_values,
    )
    # print("use matrix to compute")
    # print(cnf_matrix)
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN) if TP+FN != 0 else 0
    TPR = div0(TP, TP+FN)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP) if TN+FP != 0 else 0
    TNR = div0(TN, TN+FP)
    # Fall out or false positive rate
    # FPR = FP/(FP+TN) if FP+TN != 0 else 0
    FPR = div0(FP, FP+TN)
    # False negative rate
    # FNR = FN/(TP+FN) if TP+FN != 0 else 0
    FNR = div0(FN, TP+FN)
    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN) if TP+FP+FN+TN != 0 else 0
    
    # group-related confusion matrix
    group_FPR_dict, group_FNR_dict = {}, {}
    group_TPR_dict, group_TNR_dict = {}, {}
    for a_idx in range(num_a_values):
        
        group_references, group_predictions = references[sensitive_attributes[:, a_idx] == 1], predictions[sensitive_attributes[:, a_idx] == 1]
        group_cnf_matrix = skm.confusion_matrix(
            y_true=group_references, 
            y_pred=group_predictions,
            labels=unique_y_values,
        )
            
        group_FP = group_cnf_matrix.sum(axis=0) - np.diag(group_cnf_matrix)  
        group_FN = group_cnf_matrix.sum(axis=1) - np.diag(group_cnf_matrix)
        group_TP = np.diag(group_cnf_matrix)
        group_TN = group_cnf_matrix.sum() - (group_FP + group_FN + group_TP)
        
        
        # # Sensitivity, hit rate, recall, or true positive rate
        # group_TPR = group_TP/(group_TP+group_FN) if group_TP+group_FN != 0 else 0
        group_TPR = div0(group_TP, group_TP+group_FN)
        # # Specificity or true negative rate
        # group_TNR = group_TN/(group_TN+group_FP) if group_TN+group_FP != 0 else 0
        group_TNR = div0(group_TN, group_TN+group_FP)
        # Fall out or false positive rate
        # group_FPR = group_FP/(group_FP+group_TN) if group_FP+group_TN != 0 else 0
        group_FPR = div0(group_FP, group_FP+group_TN)
        # False negative rate
        # group_FNR = group_FN/(group_TP+group_FN) if group_TP+group_FN != 0 else 0
        group_FNR = div0(group_FN, group_TP+group_FN)
        # # Overall accuracy
        # group_ACC = (group_TP+group_TN)/(group_TP+group_FP+group_FN+group_TN) if group_TP+group_FP+group_FN+group_TN != 0 else 0
        
        # save data
        group_FPR_dict[a_idx] = group_FPR
        group_FNR_dict[a_idx] = group_FNR
        group_TPR_dict[a_idx] = group_TPR
        group_TNR_dict[a_idx] = group_TNR

    overall_metric_rates = {
        'overall_fpr_dict': FPR,
        'overall_fnr_dict': FNR,
        'overall_tpr_dict': TPR,
        'overall_tnr_dict': TNR,
    }
    group_metric_rates = {
        'group_fpr_dict': group_FPR_dict,
        'group_fnr_dict': group_FNR_dict,
        'group_tpr_dict': group_TPR_dict,
        'group_tnr_dict': group_TNR_dict,
    } 
    
    return group_metric_rates, overall_metric_rates

class FairClassificationMetrics(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="difference of equalized offs",
            citation="None",
            inputs_description="None",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                    "scores": datasets.Sequence(datasets.Value("float32")),
                    "sensitive_attributes": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            reference_urls=[],
        )

    def add_batch(self, *, predictions=None, references=None, scores=None, sensitive_attributes=None):
        """Add a batch of predictions and references for the metric's stack.
        Args:
            predictions (list/array/tensor, optional): Predictions.
            references (list/array/tensor, optional): References.
            scores (list/array/tensor, optional): scores.
            sensitive_attributes (list/array/tensor, optional): sensitive_attributes.
        """
        batch = {
            "predictions": predictions, 
            "references": references, 
            "scores": scores,
            "sensitive_attributes": sensitive_attributes,
        }

        batch = self.info.features.encode_batch(batch)
        if self.writer is None:
            self._init_writer()
        try:
            self.writer.write_batch(batch)
        except pa.ArrowInvalid as e:
            match = re.match(r"Column 1 named references expected length (\d+) but got length (\d+)", str(e))
            if match is not None:
                error_msg = (
                    f"Mismatch in the number of predictions ({match.group(1)}), references ({match.group(2)}), and sensitive_attributes ({match.group(3)}"
                )
            else:
                # lists - summarize long lists similarly to NumPy
                # arrays/tensors - let the frameworks control formatting
                def summarize_if_long_list(obj):
                    if not type(obj) == list or len(obj) <= 6:
                        return f"{obj}"

                    def format_chunk(chunk):
                        return ", ".join(repr(x) for x in chunk)

                    return f"[{format_chunk(obj[:3])}, ..., {format_chunk(obj[-3:])}]"

                error_msg = (
                    f"Predictions and/or references don't match the expected format.\n"
                    f"Expected format: {self.features},\n"
                    f"Input predictions: {summarize_if_long_list(predictions)},\n"
                    f"Input references: {summarize_if_long_list(references)},\n"
                    f"Input scores: {summarize_if_long_list(scores)},\n"
                    f"Input sensitive_attributes: {summarize_if_long_list(sensitive_attributes)}"
                )
            raise ValueError(error_msg) from None

    def add(self, *, prediction=None, reference=None, score=None, sensitive_attribute=None):
        """Add one prediction and reference for the metric's stack.
        
        Args:
            prediction (list/array/tensor, optional): Prediction.
            reference (list/array/tensor, optional): Reference.
            score (list/array/tensor, optional): scores.
            sensitive_attribute (list/array/tensor, optional): sensitive_attribute.
        """
        example = {
            "predictions": prediction, 
            "references": reference,
            "scores": score,
            "sensitive_attributes": sensitive_attribute,
        }

        example = self.info.features.encode_example(example)
        if self.writer is None:
            self._init_writer()
        try:
            self.writer.write(example)
        except pa.ArrowInvalid:
            raise ValueError(
                f"Prediction and/or reference don't match the expected format.\n"
                f"Expected format: {self.features},\n"
                f"Input predictions: {prediction},\n"
                f"Input references: {reference},\n"
                f"Input scores: {score},\n"
                f"Input sensitive_attributes: {sensitive_attribute}"
            ) from None

    def compute(self, *, predictions=None, references=None, scores=None, sensitive_attributes=None, **kwargs) -> Optional[dict]:
        """Compute the metrics.
        """
        if predictions is not None:
            self.add_batch(
                predictions=predictions, 
                references=references,
                scores=scores,
                sensitive_attributes=sensitive_attributes,
            )
        self._finalize()

        self.cache_file_name = None
        self.filelock = None

        if self.process_id == 0:
            self.data.set_format(type=self.info.format)

            predictions = self.data["predictions"]
            references = self.data["references"]
            scores = self.data["scores"]
            sensitive_attributes = self.data["sensitive_attributes"]

            with temp_seed(self.seed):
                output = self._compute(
                    predictions=predictions, 
                    references=references,
                    scores=scores,
                    sensitive_attributes=sensitive_attributes,
                    **kwargs
                )

            if self.buf_writer is not None:
                self.buf_writer = None
                del self.data
                self.data = None
            else:
                # Release locks and delete all the cache files. Process 0 is released last.
                for filelock, file_path in reversed(list(zip(self.filelocks, self.file_paths))):
                    logger.info(f"Removing {file_path}")
                    del self.data
                    self.data = None
                    del self.writer
                    self.writer = None
                    os.remove(file_path)
                    filelock.release()

            return output
        else:
            return None

    def _compute(self, predictions, references, scores, sensitive_attributes, sample_weight=None):
        ''' Actual Implementation of compute metrics
        '''
        # test whether the references is binary labels
        is_binary_label = len(set(references)) == 2
        
        # get cardinality of Y and A
        unique_y_values = _get_unique_labels(np.vstack((predictions, references)))
        num_a_values = np.asarray(sensitive_attributes).shape[1]

        # get accuracy
        accuracy = accuracy_score(
            y_true=references, 
            y_pred=predictions, 
            sample_weight=sample_weight,
        )
        
        # get f1
        f1 = f1_score(
            y_true=references, 
            y_pred=predictions, 
            average='binary' if is_binary_label else 'micro', 
            sample_weight=sample_weight,
        )

        # get macro f1
        macro_f1 = f1_score(
            y_true=references, 
            y_pred=predictions, 
            average='binary' if is_binary_label else 'macro', 
            sample_weight=sample_weight,
        )

        # get balanced accuracy
        balanced_accuracy = balanced_accuracy_score(
            y_true=references, 
            y_pred=predictions,
        )
        
        # get roc_auc
        # NOTE: average will be ignored when y_true is binary.
        roc_auc = roc_auc_score(
            y_true=references,
            y_score=np.array(scores)[:, 1] if is_binary_label else scores,
            average='macro', # 'macro' if is_binary_label else 'weighted',
            multi_class='raise' if is_binary_label else 'ovr',
            sample_weight=sample_weight,
        )

        #####################################################
        ######### compute EO-based fairness metrics #########
        #####################################################
        group_metric_rates, overall_metric_rates = compute_deo_for_multi_label_loop(
            predictions=np.asarray(predictions),
            references=np.asarray(references),
            sensitive_attributes=np.asarray(sensitive_attributes),
        )

        # by group metrics (could be extended when a is not binary)
        fprs_diff, fnrs_diff = [], []
        tprs_diff, tnrs_diff = [], []
        for y_idx in unique_y_values:
            for a_idx in range(num_a_values):
                fprs_diff.append(np.abs( group_metric_rates['group_fpr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_fpr_dict'][y_idx] ))
                fnrs_diff.append(np.abs( group_metric_rates['group_fnr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_fnr_dict'][y_idx] ))
                tprs_diff.append(np.abs( group_metric_rates['group_tpr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_tpr_dict'][y_idx] ))
                tnrs_diff.append(np.abs( group_metric_rates['group_tnr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_tnr_dict'][y_idx] ))
        
        FPR_gap = np.sum(fprs_diff)
        FNR_gap = np.sum(fnrs_diff)
        TPR_gap = np.sum(tprs_diff)
        TNR_gap = np.sum(tnrs_diff)
        
        rms_FPR_gap = rms_diff(np.array(fprs_diff))
        rms_FNR_gap = rms_diff(np.array(fnrs_diff))
        rms_TPR_gap = rms_diff(np.array(tprs_diff))
        rms_TNR_gap = rms_diff(np.array(tnrs_diff))

        assert np.abs(FPR_gap - TNR_gap) <= 1e-6
        assert np.abs(FNR_gap - TPR_gap) <= 1e-6 

        assert np.abs(rms_FPR_gap - rms_TNR_gap) <= 1e-6
        assert np.abs(rms_FNR_gap - rms_TPR_gap) <= 1e-6

        # NOTE:by max metrics (the current implementation only works when a is binary)
        if num_a_values == 2:
            # by max rms
            fprs_diff_by_max, fnrs_diff_by_max = [], []
            tprs_diff_by_max, tnrs_diff_by_max = [], []
            for y_idx in range(len(unique_y_values)):
                # fpr
                fpr_diff = group_metric_rates['group_fpr_dict'][(y_idx, 0)] \
                    - group_metric_rates['group_fpr_dict'][(y_idx, 1)]
                fprs_diff_by_max.append(fpr_diff)
                # fnr
                fnr_diff = group_metric_rates['group_fnr_dict'][(y_idx, 0)] \
                    - group_metric_rates['group_fnr_dict'][(y_idx, 1)]
                fnrs_diff_by_max.append(fnr_diff)
                # tpr
                tpr_diff = group_metric_rates['group_tpr_dict'][(y_idx, 0)] \
                    - group_metric_rates['group_tpr_dict'][(y_idx, 1)]
                tprs_diff_by_max.append(tpr_diff)
                # tnr
                tnr_diff = group_metric_rates['group_tnr_dict'][(y_idx, 0)] \
                    - group_metric_rates['group_tnr_dict'][(y_idx, 1)]
                tnrs_diff_by_max.append(tnr_diff)

            fprs_diff_by_max = np.array(fprs_diff_by_max)
            fnrs_diff_by_max = np.array(fnrs_diff_by_max)
            tprs_diff_by_max = np.array(tprs_diff_by_max)
            tnrs_diff_by_max = np.array(tnrs_diff_by_max)

            rms_FPR_gap_by_max = rms_diff(fprs_diff_by_max)
            rms_FNR_gap_by_max = rms_diff(fnrs_diff_by_max)
            rms_TPR_gap_by_max = rms_diff(tprs_diff_by_max)
            rms_TNR_gap_by_max = rms_diff(tnrs_diff_by_max)

            assert np.abs(rms_FPR_gap_by_max - rms_TNR_gap_by_max) <= 1e-6
            assert np.abs(rms_FNR_gap_by_max - rms_TPR_gap_by_max) <= 1e-6

        # cnf matrix can be used for validation of fairness metrics
        # group_metric_rates_cnf_mat, overall_metric_rates_cnf_mat = compute_deo_for_multi_label_cnf_matrix(
        #     predictions=np.asarray(predictions),
        #     references=np.asarray(references),
        #     sensitive_attributes=np.asarray(sensitive_attributes),
        # )
        
        metrics = {
            'accuracy': accuracy, 
            'f1': f1,
            'macro_f1': macro_f1,
            'balanced_accuracy': balanced_accuracy,
            'roc_auc': roc_auc,
            'FPR_gap': FPR_gap,
            'FNR_gap': FNR_gap,
            'EO_gap': FPR_gap+FNR_gap,
            'rms_FPR_gap': rms_FPR_gap,
            'rms_FNR_gap': rms_FNR_gap,
            'rms_EO_gap':rms_FPR_gap+rms_FNR_gap,
        }

        if num_a_values == 2:
            # add by max metrics to final evaluation for comparision
            metrics['rms_FPR_gap_by_max'] = rms_FPR_gap_by_max
            metrics['rms_FNR_gap_by_max'] = rms_FNR_gap_by_max
            metrics['rms_EO_gap_by_max'] = rms_FPR_gap_by_max+rms_FNR_gap_by_max
        
        data_arrays = {
            "predictions": predictions, 
            "references": references, 
            "scores": scores, 
            "sensitive_attributes": sensitive_attributes,
        }
        return metrics, data_arrays