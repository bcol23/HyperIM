import sys
sys.path.append('..') 

import torch as th
from tqdm import tqdm

from .evalu import precision_k, ndcg_k
from .hyperop import project_hyp_vec


def train_EuclideanIM(epoch, net, train_data_loader, loss, optim):
    for e in tqdm(range(1, epoch + 1), desc='train epoch'):
        train_loss = 0
        p1, p3, p5 = 0, 0, 0
        ndcg1, ndcg3, ndcg5 = 0, 0, 0

        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_data_loader), desc='batch'):

            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            optim.zero_grad()
            output = net(X_batch)
            l = loss(output, y_batch)
            l.backward()
            optim.step()

            with th.no_grad():
                train_loss += l.item()

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

        print('epoch', e)
        print('P@1\t%.3f\t\tP@3\t%.3f\t\tP@5\t%.3f' %(p1, p3, p5))
        print('nDCG@1\t%.3f\t\tnDCG@3\t%.3f\t\tnDCG@5\t%.3f' %(ndcg1, ndcg3, ndcg5))


def train_HyperIM(epoch, net, train_data_loader, loss, optim):
    for e in tqdm(range(1, epoch + 1), desc='train epoch'):
        train_loss = 0
        p1, p3, p5 = 0, 0, 0
        ndcg1, ndcg3, ndcg5 = 0, 0, 0

        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_data_loader), desc='batch'):
            with th.no_grad():
                net.word_embed.data = project_hyp_vec(net.word_embed)
                net.label_embed.data = project_hyp_vec(net.label_embed)

            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            optim.zero_grad()
            output = net(X_batch)
            l = loss(output, y_batch)
            l.backward()
            optim.step()

            with th.no_grad(): 
                train_loss += l.item()

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

        print('epoch', e)
        print('P@1\t%.3f\t\tP@3\t%.3f\t\tP@5\t%.3f' %(p1, p3, p5))
        print('nDCG@1\t%.3f\t\tnDCGp@3\t%.3f\t\tnDCG@5\t%.3f' %(ndcg1, ndcg3, ndcg5))
