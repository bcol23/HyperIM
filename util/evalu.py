from datetime import datetime
import sys
sys.path.append('..') 

import torch as th
import pandas as pd
from tqdm import tqdm

from params import *


def precision_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]
    
    precision = []
    for _k in k:
        p = 0
        for i in range(batch_size):
            p += label[i, pred[i, :_k]].mean().item()
        precision.append(p*100/batch_size)
    
    return precision


def ndcg_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]
    
    ndcg = []
    for _k in k:
        score = 0
        rank = th.log2(th.arange(2, 2 + _k, dtype=default_dtype, device=cuda_device))
        for i in range(batch_size):
            l = label[i, pred[i, :_k]]
            n = l.sum().item()
            if(n == 0):
                continue
            
            dcg = (l/rank).sum().item()
            label_count = label[i].sum().item()
            norm = 1 / th.log2(th.arange(2, 2 + min(_k, label_count), dtype=default_dtype))
            norm = norm.sum().item()
            score += dcg/norm
            
        ndcg.append(score*100/batch_size)
    
    return ndcg


def evaluate(net, test_data_loader):
    p1, p3, p5 = 0, 0, 0
    ndcg1, ndcg3, ndcg5 = 0, 0, 0
    
    with th.no_grad():
        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(test_data_loader), desc='evaluating'):

            _batch_size = X_batch.shape[0]
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            output = net(X_batch)
            pred = output.topk(k=5)[1]
            
            _p1, _p3, _p5 = precision_k(pred, y_batch, k=[1, 3, 5])
            p1 += _p1
            p3 += _p3
            p5 += _p5

            _ndcg1, _ndcg3, _ndcg5 = ndcg_k(pred, y_batch, k=[1, 3, 5])
            ndcg1 += _ndcg1
            ndcg3 += _ndcg3
            ndcg5 += _ndcg5
    
    batch_idx += 1
    p1 /= batch_idx
    p3 /= batch_idx
    p5 /= batch_idx
    ndcg1 /= batch_idx
    ndcg3 /= batch_idx
    ndcg5 /= batch_idx
    
    print('P@1\t%.3f\t\tP@3\t%.3f\t\tP@5\t%.3f' %(p1, p3, p5))
    print('nDCG@1\t%.3f\t\tnDCG@3\t%.3f\t\tnDCG@5\t%.3f' %(ndcg1, ndcg3, ndcg5))
    
    if(if_log):
        log_columns = ['P@1', 'P@3', 'P@5', 'nDCGP@1', 'nDCG@3', 'nDCG@5']
        log = pd.DataFrame([[p1, p3, p5, ndcg1, ndcg3, ndcg5]], columns=log_columns)
        log.to_csv('./log/result-dim-' + str(embed_dim) + '-' + str(datetime.now()) + '.csv', 
                   encoding='utf-8', index=False)
