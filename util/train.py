import sys
sys.path.append('..') 

import torch as th
from tqdm import tqdm

from .hyperop import project_hyp_vec


def train(epoch, net, loss, optim, if_hyper=True, if_neg_samp=False, train_data_loader=None, 
          data_path='./data/sample', train_batch_size=50, word_num=500):
    
    if(train_data_loader == None):
        train_data_loader = load_train_data(data_path, train_batch_size, word_num)
    
    for e in tqdm(range(1, epoch + 1), desc='train epoch'):
        train_loss = 0

        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_data_loader), desc='batch'):
            if(if_hyper):
                with th.no_grad():
                    net.word_embed.data = project_hyp_vec(net.word_embed)
                    net.label_embed.data = project_hyp_vec(net.label_embed)
                
            if(if_neg_samp):
                label = y_batch.long()
                label = list(set(label.nonzero().numpy()[:, -1]))
                label.sort()

                X_batch = X_batch.cuda()
                y_batch = y_batch[:, label].cuda()

                optim.zero_grad()
                l = loss(net(X_batch, label), y_batch)
            else:
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

                optim.zero_grad()
                output = net(X_batch)
                l = loss(output, y_batch)
            
            train_loss += l
            l.backward()
            optim.step()
        
        print('epoch', e, '\tloss', train_loss.item())
        th.cuda.empty_cache()
