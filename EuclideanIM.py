import torch as th
import torch.nn as nn

from net.EuclideanIM import EuclideanIM
from util import train, evalu, data


default_dtype = th.float64
th.set_default_dtype(default_dtype)

if th.cuda.is_available():
    cuda_device = th.device('cuda:0')
    th.cuda.set_device(device=cuda_device)
else:
    raise Exception('No CUDA device found.')
    
data_path = './data/sample/'

# for the sample
label_num = 103
vocab_size = 50000
word_num = 200

if_gru = True # otherwise use rnn
if_log = True # log result

epoch = 1
embed_dim = 10

train_batch_size = 50
test_batch_size = 50
lr = 1e-4


if(__name__ == '__main__'):
    # use pre-trained embed if avalible
    word_embed = th.rand(vocab_size, embed_dim)
    label_embed = th.rand(label_num, embed_dim)

    net = EuclideanIM(word_num, word_embed, label_embed, hidden_size=embed_dim, if_gru=if_gru)
    net.to(cuda_device)

    loss = nn.BCEWithLogitsLoss()
    optim = th.optim.Adam(net.parameters(), lr=lr)

    train_data_loader, test_data_loader = data.load_data(data_path, train_batch_size, test_batch_size, word_num)

    train.train(epoch, net, loss, optim, if_hyper=False, if_neg_samp=False, train_data_loader=train_data_loader)
    evalu.evaluate(net, if_log=if_log, test_data_loader=test_data_loader)
    