''' Script to evaluate trained CLAD 
'''

import torch as T
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import argparse
from data.load_data import get_data
from model.model import CLOSRMLP
from util.checkpoint import load_checkpoint
from util.features import get_features
import numpy as np
from pprint import pprint
from util.metrics import mean_auroc, classwise_fpr_at_recall, fpr_at_recall, compute_mean_ap_and_pr_auc
from util.distance import get_centroids, calc_class_sims_chunked
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # data config
    parser.add_argument('--data_path', type=str, default='data/lycos.csv', help='path to dataset')
    parser.add_argument('--drop_cols', type=str, default='flow_id,src_addr,src_port,dst_addr,dst_port,ip_prot,timestamp', help='columns to drop from dataset')
    parser.add_argument('--sample_thres', type=int, default=100, help='maximum number before exclusion as a zero day attack')
    parser.add_argument('--split_seed', type=int, default=39058032, help='seed for train test split')
    
    # model config
    parser.add_argument('--d_out', type=int, default=64, help='model output dimensionality')
    parser.add_argument('--n_classes', type=int, default=12, help='number of classes in dataset')
    parser.add_argument('--neurons', type=str, default='1024,1024,1024', help='neurons in each mlp block')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--residual', type=bool, default=True, help='Whether to use residual connections in mlp')
    parser.add_argument('--checkpoint_path', type=str, default='weights/closr.pt.tar', help='path to saved weights')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--chunk_size', type=int, default=1024, help='chunk size for getting features during inference')
    
    opt = parser.parse_args()
    
    # parse drop cols into list
    drop_cols = opt.drop_cols.split(',')
    opt.drop_cols = list([])
    for c in drop_cols:
        opt.drop_cols.append(c)
        
    # parse neurons into list
    neurons = opt.neurons.split(',')
    opt.neurons = list([])
    for n in neurons:
        opt.neurons.append(int(n))

    return opt

def load_data(opt):
    x_train, y_train, _, _, x_test, y_test, x_zd, y_zd = get_data(
        data_path = opt.data_path, 
        target = 'label', 
        drop = opt.drop_cols, 
        class_zero = 'benign', 
        sample_thres = opt.sample_thres,
        split_seed = opt.split_seed,
        test_ratio = 0.5,
        val_ratio = 0.0,
    )
    x_train = T.tensor(x_train, dtype = T.float32, device = opt.device)
    x_zd = T.tensor(x_zd, dtype = T.float32, device = opt.device)
    x_test = T.tensor(x_test, dtype = T.float32, device = opt.device)
    
    y_train = T.tensor(y_train, dtype = T.int64, device = opt.device)
    y_zd = T.tensor(y_zd, dtype = T.int64, device = opt.device)
    y_test = T.tensor(y_test, dtype = T.int64, device = opt.device)

    return x_train, y_train, x_test, y_test, x_zd, y_zd

def load_model(opt):
    # get model
    model = CLOSRMLP(
        d_in = 72,
        n_classes = opt.n_classes,
        d_out = opt.d_out,
        neurons = opt.neurons,
        dropout = opt.dropout,
        residual = opt.residual,
    )
    
    # load weights
    model, _, _, _, _ = load_checkpoint(
        opt.checkpoint_path,
        model,
    )
    model = model.to(opt.device)
    model.eval()
    return model

@T.no_grad()
def closr_eval(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    x_zd: Tensor,
    y_zd: Tensor,
    opt,
) -> dict:

    # number of zd classes 
    num_known = T.unique(y_test).numel()
    
    # number of known classes
    if x_zd is not None and y_zd is not None:
        x_test = T.cat((x_test, x_zd), dim = 0)
        y_test = T.cat((y_test, y_zd), dim = 0)
    
    # get embeddings
    train_features, train_labels = get_features(
        model = model,
        x_data = x_train,
        y_data = y_train,
        chunk_size = opt.chunk_size,
        move_to_cpu = False,
    )
    
    test_features, test_labels = get_features(
        model = model,
        x_data = x_test,
        y_data = y_test,
        chunk_size = opt.chunk_size,
        move_to_cpu = False,
    )
    
    # get class centroids
    centroids = F.normalize(get_centroids(train_features, train_labels), dim = -1)
    
    # make closed set predictions
    test_sims = calc_class_sims_chunked(
        centroids = centroids,
        embeddings = test_features,
        chunk_size = opt.chunk_size
    )
    
    # closed set predictions
    test_preds = np.argmax(test_sims.cpu().detach().numpy(), axis =1)
    
    # open set predictions
    test_probs = T.softmax(test_sims, dim = -1)
    osr_scores = T.sum(T.pow(test_sims,2) * test_probs ,dim =-1).cpu().detach().numpy()
    
    # get metrics
    test_labels = test_labels.cpu().detach().numpy()
    metrics = {}
    
    # open auc
    metrics['closed_set_acc'] = np.mean(test_labels[test_labels < num_known] == test_preds[test_labels < num_known])
    
    ood_labels = test_labels.copy()
    ood_labels[ood_labels < num_known] = 0
    ood_labels[ood_labels != 0] = 1
    metrics['open_set_auc'] = roc_auc_score(ood_labels, -1 * osr_scores)
    metrics['open_auc'] = metrics['closed_set_acc'] * metrics['open_set_auc']
    metrics['fpr_95'] = fpr_at_recall(ood_labels, -1*osr_scores)
    metrics['mean_fpr_95'] = classwise_fpr_at_recall(-1*osr_scores, test_labels, class_thres=num_known)[-1]
    
    # closed set metrics
    mal_mask = (test_labels > 0) & (test_labels < num_known)
    mal_labels = test_labels.copy()
    mal_labels = test_labels[mal_mask] - 1
    mal_preds = np.argmax(test_sims[mal_mask][:,1:].cpu().detach().numpy(), axis =-1)
    
    metrics['mean_recall'] = balanced_accuracy_score(test_labels[test_labels < num_known], test_preds[test_labels < num_known])
    metrics['mal_recall'] = balanced_accuracy_score(mal_labels, mal_preds)
    metrics['macro_f1'] = f1_score(test_labels[test_labels < num_known], test_preds[test_labels < num_known], average = 'macro')
    metrics['mal_macro_f1'] = f1_score(mal_labels, mal_preds, average = 'macro')
    metrics['closed_set_mean_auroc'] = mean_auroc(scores = test_probs[ood_labels == 0][:,0], y_true= test_labels[ood_labels == 0])
    metrics['closed_set_auroc'] = mean_auroc(scores = test_probs[ood_labels == 0][:,0], y_true= (test_labels[ood_labels == 0] > 0).astype(int))
    
    metrics['fp_rate'] = 1 - ((np.sum((test_labels == 0) & (test_preds == 0)) / (np.sum(test_labels == 0))))
    
    ap_score, pr_score = compute_mean_ap_and_pr_auc(test_labels, test_probs, num_known)
    metrics['mean_average_precision_score'] = ap_score
    metrics['mean_precision_recall_auc'] = pr_score
    
    return metrics

def main():
    opt = parse_option()

    # get data
    x_train, y_train, x_test, y_test, x_zd, y_zd = load_data(opt)
    
    # get model
    model = load_model(opt)
    
    # eval model
    metrics = closr_eval(
        model = model,
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test,
        x_zd = x_zd,
        y_zd = y_zd,
        opt = opt,
    )
    
    pprint(metrics)
    
if __name__ == '__main__':
    main()