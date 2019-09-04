import sys
sys.path.append('..') 

import torch as th
import torch.nn as nn
import geoopt as gt

from util.hyperop import *


class hyperDense(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(hyperDense, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        k = (1 / in_features)**0.5
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(in_features, out_features).uniform_(-k, k))
        self.b = gt.ManifoldParameter(gt.ManifoldTensor(out_features).zero_())
    
    
    def forward(self, inputs):
        hyp_b = exp_map_zero(self.b)
        
        wx = mob_mat_mul(self.w, inputs)
        
        return mob_add(wx, hyp_b)
    

class hyperRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, default_dtype=th.float64):
        super(hyperRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.default_dtype = default_dtype
        
        k = (1 / hidden_size)**0.5
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.u = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.b = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        
        
    def transition(self, x, h):
        W_otimes_h = mob_mat_mul(self.w, h)
        U_otimes_x = mob_mat_mul(self.u, x)
        Wh_plus_Ux = mob_add(W_otimes_h, U_otimes_x)
        
        return mob_add(Wh_plus_Ux, self.b)
    
    
    def init_rnn_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros((batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device)
    
    
    def forward(self, inputs):
        hidden = self.init_rnn_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.transition(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)
    
    
class GRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        k = (1 / hidden_size)**0.5
        self.w_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.w_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.w_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.u_z = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.u_r = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.u_h = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.b_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        self.b_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        self.b_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        
    
    def transition(self, W, h, U, x, hyp_b):
        W_otimes_h = mob_mat_mul(W, h)
        U_otimes_x = mob_mat_mul(U, x)
        Wh_plus_Ux = mob_add(W_otimes_h, U_otimes_x)
        
        return mob_add(Wh_plus_Ux, hyp_b)
    
    
    def forward(self, hyp_x, hidden):
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = th.sigmoid(log_map_zero(z))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = th.sigmoid(log_map_zero(r))

        r_point_h = mob_pointwise_prod(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_r, hyp_x, self.b_h)
        # h_tilde = th.tanh(log_map_zero(h_tilde)) # non-linearity

        minus_h_oplus_htilde = mob_add(-hidden, h_tilde)
        new_h = mob_add(hidden, mob_pointwise_prod(minus_h_oplus_htilde, z))
        
        return new_h
    
    
class hyperGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, default_dtype=th.float64):
        super(hyperGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.default_dtype = default_dtype
        
        self.gru_cell = GRUCell(input_size, hidden_size)
    
    
    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros((batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device)
    
    
    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)
    