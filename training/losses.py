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

class HardestRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(HardestRankingLoss, self).__init__()
        self.margin=margin
        self.relu=nn.ReLU()

    def forward(self, images, captions):
        assert images.shape==captions.shape and len(images.shape)==2
        images=images/torch.norm(images,dim=1,keepdim=True)
        captions=captions/torch.norm(captions,dim=1,keepdim=True)        
        num_samples=len(images)

        similarity_scores = torch.mm( images, captions.transpose(1,0) ) # [I x C]

        cost_images= self.margin + similarity_scores - similarity_scores.diag().view((num_samples,1))
        cost_images.fill_diagonal_(0)
        cost_images=self.relu(cost_images)
        cost_images,_=torch.max(cost_images, dim=1)
        cost_images=torch.mean(cost_images)

        cost_captions= self.margin + similarity_scores.transpose(1,0) - similarity_scores.diag().view((num_samples,1))
        cost_captions.fill_diagonal_(0)
        cost_captions=self.relu(cost_captions)
        cost_captions,_=torch.max(cost_captions, dim=1)
        cost_captions=torch.mean(cost_captions)        

        cost= cost_images+cost_captions      
        return cost         