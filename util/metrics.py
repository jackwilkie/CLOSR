''' Evaluation metrics
'''

from torch import Tensor
import numpy as np
from numpy import ndarray
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_recall_curve, auc, average_precision_score
from typing import Optional

# -- auroc
def mean_auroc(
    scores,
    y_true,
    return_class_level: bool = False,
    include_lower_thres: bool = True,
):
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()
    
    if isinstance(scores, Tensor):
        scores = scores.cpu().detach().numpy()
    
    roc_scores = []
    
    for c in np.unique(y_true):
        if c == 0:
            # no auroc for benign data
            continue
        
        # get attack class and benign traffic
        class_mask = (y_true == 0) | (y_true == c)
        y = y_true[class_mask]  
        y[y>0] = 1

        x = scores[class_mask]
        roc = roc_auc_score(y,x)
        if include_lower_thres:
            roc_scores.append(max(roc, 1-roc))
        else:
            roc_scores.append(roc) 
    
    roc_scores.append(np.mean(roc_scores))
    
    if return_class_level:
        return roc_scores
    else:
        return roc_scores[-1]

def balanced_auroc(scores, labels, return_class_level: bool = False):
    class_auroc = mean_auroc(scores = scores, y_true = labels, return_class_level= True)
    
    if return_class_level:
        return class_auroc
    else:
        return class_auroc[:-1]

# -- fpr@95
def fpr_at_recall(y_true, y_score, recall_level=0.95):
    y_true = y_true.copy()
        
    y_true = np.asarray(y_true) > 0
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    tps = np.cumsum(y_true)
    fps = np.cumsum(~y_true)
    recall = tps / tps[-1]
    
    valid = recall >= recall_level
    if not np.any(valid):
        return 0.0  # or np.nan

    cutoff = np.argmax(recall >= recall_level)
    fpr = fps[cutoff] / np.sum(~y_true)
    return fpr

def classwise_fpr_at_recall(
    scores,
    y_true,
    recall_level = 0.95,
    class_thres = None,
):  
    
    if isinstance(scores, Tensor):
        scores = scores.cpu().detach().numpy()
        
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()        
    
    y_true = y_true.copy()
    
    if class_thres is not None:
        y_true[y_true < class_thres] = 0
    
    fpr_scores = []
    
    for c in np.unique(y_true):
        if c == 0:
            # no fpr for benign data
            continue
        
        # get attack class and benign traffic
        class_mask = (y_true == 0) | (y_true == c)
        y = y_true[class_mask]  
        y[y>0] = 1

        x = scores[class_mask]
        fpr_scores.append(fpr_at_recall(y,x,recall_level))
    
    fpr_scores.append(np.mean(fpr_scores))
    return fpr_scores

# -- precision-recall auc
def compute_mean_ap_and_pr_auc(test_labels, test_probs, num_known, reference_class=0):

    ap_scores = []
    pr_auc_scores = []

    for c in range(num_known):
        if c == reference_class:
            continue

        # Filter: keep only samples from class c and reference_class
        mask = np.isin(test_labels, [reference_class, c])
        y_true = (test_labels[mask] == c).astype(int)  # binary labels: 1 for class c, 0 for reference_class
        y_score = test_probs[mask][:, c]               # predicted probability for class c

        # Average precision (step-wise)
        ap = average_precision_score(y_true, y_score)
        ap_scores.append(ap)

        # PR AUC (trapezoidal, with deduplication of recall)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        recall_unique, idx = np.unique(recall, return_index=True)
        pr_auc = auc(recall_unique, precision[idx])
        pr_auc_scores.append(pr_auc)

    return np.mean(ap_scores),np.mean(pr_auc_scores)
