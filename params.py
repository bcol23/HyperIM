import torch as th

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
word_num = 120

# use the embed dim xd_ball(>1) workaround to avoid numeric errors,
# otherwise don't use
d_ball = 0 

if_gru = True # otherwise use rnn
if_log = True # log result

epoch = 1
embed_dim = 2 # xd_ball if d_ball > 1

train_batch_size = 50
test_batch_size = 50
lr = 1e-4
