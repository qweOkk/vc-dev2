# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the linear model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels), print(x.shape, labels.shape)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        labels = labels.reshape(-1)
        # print([x].shape)
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
    
class TDNN(nn.Module):
        
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)
        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )
        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        if self.dropout_p:
            x = self.drop(x)
        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)
        return x

class XVector(nn.Module):
    def __init__(self, input_dim=512, output_dim = 1500, nOut=512):
        super(XVector, self).__init__()
        # simply take mean operator / no additional parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nOut = nOut
        self.module = nn.Sequential(
            TDNN(input_dim=self.input_dim, output_dim=self.input_dim, context_size=5, dilation=1),
            TDNN(input_dim=self.input_dim, output_dim=self.input_dim, context_size=3, dilation=2),
            TDNN(input_dim=self.input_dim, output_dim=self.input_dim, context_size=3, dilation=3),
            TDNN(input_dim=self.input_dim, output_dim=self.input_dim, context_size=1, dilation=1),
            TDNN(input_dim=self.input_dim, output_dim=self.output_dim, context_size=1, dilation=1),
        )
        p_dropout = 0.1
        self.fc1 = nn.Linear(2 * self.output_dim, self.input_dim)
        self.bn_fc1 = nn.BatchNorm1d(self.input_dim, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.bn_fc2 = nn.BatchNorm1d(self.input_dim, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(self.input_dim, self.nOut)

    def forward(self, feature):
        x = self.module(feature)
        # B x T X D --> B x D x T
        x = x. transpose(1, 2)   
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x