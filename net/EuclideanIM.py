import torch as th
import torch.nn as nn
import torch.nn.functional as F


class EuclideanIM(nn.Module):
    
    def __init__(self, feature_num, word_embed, label_embed, hidden_size=5, if_gru=True, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        
        self.word_embed = nn.Embedding(word_embed.shape[0], word_embed.shape[1], padding_idx=1).from_pretrained(
                        word_embed, freeze=False)
        self.label_embed = nn.Parameter(label_embed)
        
        if(if_gru):
            self.rnn = nn.GRU(word_embed.shape[1], hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(word_embed.shape[1], hidden_size, batch_first=True)
        
        self.dense_1 = nn.Linear(feature_num, int(feature_num/2))
        self.dense_2 = nn.Linear(int(feature_num/2), 1)
    
    
    def forward(self, x, label=None):
        word_embed = self.word_embed(x)
        encode = self.rnn(word_embed)[0]
        encode = encode.unsqueeze(dim=2)
        
        if(label == None):
            encode = encode.expand(-1, -1, self.label_embed.shape[0], -1)
            interaction = ((encode - self.label_embed.expand_as(encode))**2).sum(dim=-1)
            
        else:
            encode = encode.expand(-1, -1, len(label), -1)
            interaction = ((encode - self.label_embed[label].expand_as(encode))**2).sum(dim=-1)
        
        interaction = interaction.squeeze(dim=-1).transpose(1, 2)
        
        out = F.relu(self.dense_1(interaction))
        out = self.dense_2(out).squeeze(dim=-1)
        
        return out
    