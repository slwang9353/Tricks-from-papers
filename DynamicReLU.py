import torch
from torch.functional import einsum
import torch.nn as nn
# import torch.einsum as einsum

class MLP(nn.Module):
    def __init__(self, widths, bn=True):
        '''widths [in_channel, ..., out_channel], with ReLU within'''
        super(MLP, self).__init__()
        layers = []
        for n in range(len(widths) - 1):
            layer_ = nn.Sequential(
                nn.Linear(widths[n], widths[n + 1]),
                nn.ReLU6(inplace=True),
            )
            layers.append(layer_)
        self.mlp = nn.Sequential(*layers)
        if bn:
            self.mlp = nn.Sequential(
                *layers,
                nn.BatchNorm1d(widths[-1])
            )
    def forward(self, x):
        return self.mlp(x)


class DynamicReLU(nn.Module):
    def __init__(self, in_channel, control_demension, k=2):
        '''channel-width weighted DynamticReLU '''
        '''Yinpeng Chen, Xiyang Dai et al., Dynamtic ReLU, arXiv preprint axXiv: 2003.10027v2'''
        super(DynamicReLU, self).__init__()
        self.in_channel = in_channel
        self.k = k
        self.control_demension = control_demension
        self.Theta = MLP([control_demension, 4 * control_demension, 2 * k * in_channel], bn=True)
    def forward(self, x, control_vector):
        _, tokens, _ = control_vector.shape
        n, _, _, _ = x.shape
        if tokens != 1:
            control_vector = torch.flatten(control_vector, 1)   # Only first token is used in original paper
        control_vector = control_vector.squeeze()
        a_defalut = torch.ones(n, self.k * self.in_channel)
        a_defalut[:, self.k * self.in_channel // 2 : ] = torch.zeros(n, self.k * self.in_channel // 2)
        theta = self.Theta(control_vector)
        theta = 2 * torch.sigmoid(theta) - 1
        a = theta[:, 0 : self.k * self.in_channel]
        a = a_defalut + a
        b = theta[:, self.k * self.in_channel : ] * 0.5
        a = a.reshape(n, self.k, self.in_channel)
        b = b.reshape(n, self.k, self.in_channel)
        # x (NCHW), a & b (N, k, C)
        x = einsum('nchw, nkc -> nchwk', x, a) + einsum('nchw, nkc -> nchwk', torch.ones_like(x), b)
        return x.max(4)[0]


