import sys
sys.path.append('..') 

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import geoopt as gt

from .hypernnxd import *
from util.hyperop import poinc_dist


class HyperIM(nn.Module):
    
    def __init__(self, feature_num, word_embed, label_embed, d_ball, hidden_size=5, if_gru=False, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.d_ball = d_ball
        
        self.word_embed = gt.ManifoldParameter(word_embed, manifold=gt.PoincareBall())
        self.label_embed = gt.ManifoldParameter(label_embed, manifold=gt.PoincareBall())
        
        if(if_gru):
            self.rnn = hyperGRU(input_size=word_embed.shape[1], hidden_size=self.hidden_size, d_ball=self.d_ball)
        else:
            self.rnn = hyperRNN(input_size=word_embed.shape[1], hidden_size=self.hidden_size, d_ball=self.d_ball)
        
        self.dense_1 = nn.Linear(feature_num, int(feature_num/2))
        self.dense_2 = nn.Linear(int(feature_num/2), 1)
    

    def forward(self, x):
        word_embed = self.word_embed[x]
        encode = self.rnn(word_embed)

        encode = encode.unsqueeze(dim=2)
        encode = encode.expand(-1, -1, self.label_embed.shape[0], -1, -1)
        
        interaction = poinc_dist(encode, self.label_embed.expand_as(encode))
        interaction = interaction.squeeze(dim=-1).sum(dim=-1).transpose(1, 2)
        
        out = F.relu(self.dense_1(interaction))
        out = self.dense_2(out).squeeze(dim=-1)
        
        return out