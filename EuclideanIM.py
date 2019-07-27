import torch.nn as nn

from net.EuclideanIM import EuclideanIM
from util import train, evalu, data
from params import *

if(__name__ == '__main__'):
    # use pre-trained embed if avalible
    word_embed = th.rand(vocab_size, embed_dim)
    label_embed = th.rand(label_num, embed_dim)
    
    net = EuclideanIM(word_num, word_embed, label_embed, hidden_size=embed_dim, if_gru=if_gru)
    net.to(cuda_device)

    loss = nn.BCEWithLogitsLoss()
    optim = th.optim.Adam(net.parameters(), lr=lr)
    
    train_data_loader, test_data_loader = data.load_data(data_path, train_batch_size, test_batch_size, word_num)
    
    train.train_EuclideanIM(epoch, net, train_data_loader, loss, optim)
    evalu.evaluate(net, test_data_loader)
    