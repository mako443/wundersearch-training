import numpy as np
import torch
import torch.nn as nn

class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s): 
        im = im/torch.norm(im, dim=1, keepdim=True)
        s = s/torch.norm(s, dim=1, keepdim=True)

        margin = self.margin
        scores = torch.mm(im, s.transpose(1, 0))
        diagonal = scores.diag()

        cost_s = torch.max(torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        cost_im = torch.max(torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return (cost_s.sum() + cost_im.sum()) / len(im) #Take mean for batch-size stability  