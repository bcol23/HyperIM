import torch.nn as nn
import geoopt as gt

from util import train, evalu, data
from params import *


if(__name__ == '__main__'):
    if(d_ball > 1):
        from net.HyperIMxd import HyperIM

        # use pre-trained embed if avalible
        word_embed = th.Tensor(vocab_size, embed_dim, d_ball)
        label_embed = th.Tensor(label_num, embed_dim, d_ball)

        net = HyperIM(word_num, word_embed, label_embed, d_ball, hidden_size=embed_dim, if_gru=if_gru)
    else:
        from net.HyperIM import HyperIM

        # use pre-trained embed if avalible    
        word_embed = th.Tensor(vocab_size, embed_dim)
        label_embed = th.Tensor(label_num, embed_dim)

        net = HyperIM(word_num, word_embed, label_embed, hidden_size=embed_dim, if_gru=if_gru)
        

    net = HyperIM(word_num, word_embed, label_embed, d_ball, hidden_size=embed_dim, if_gru=if_gru)
    net.to(cuda_device)

    loss = nn.BCEWithLogitsLoss()
    # optim = gt.optim.RiemannianSGD(net.parameters(), lr=lr, momentum=0.9, stabilize=1)
    optim = gt.optim.RiemannianAdam(net.parameters(), lr=lr)

    train_data_loader, test_data_loader = data.load_data(data_path, train_batch_size, test_batch_size, word_num)

    train.train_HyperIM(epoch, net, train_data_loader, loss, optim)
    evalu.evaluate(net, test_data_loader)
