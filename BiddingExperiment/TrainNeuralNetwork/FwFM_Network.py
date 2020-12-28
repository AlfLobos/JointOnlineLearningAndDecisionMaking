import torch
import torch.nn as nn
import torch.nn.functional as F

## This function restuns a vector [size_embs * num_embs,1] and is made to be used with the function
## BCEWithLogitsLoss() in Pytorch
class FwFM_Logits(nn.Module):
    def __init__(self, size_vocab, size_embs, num_embs, mask_to_use, scale_grad_by_freq= False):
        super(FwFM_Logits, self).__init__()
        self.emb = 0
        if torch.cuda.is_available():
            self.emb = nn.Embedding(size_vocab, size_embs, max_norm = 1, \
                scale_grad_by_freq = scale_grad_by_freq).cuda()
        else:
            self.emb = nn.Embedding(size_vocab, size_embs, max_norm = 1, \
                scale_grad_by_freq = scale_grad_by_freq)
        self.num_embstimesSize =  size_embs * num_embs
        self.linearCat = nn.Linear(self.num_embstimesSize, 1)
        self.linearTwoWay = nn.Linear(num_embs * num_embs, 1)
        self.mask_to_use = mask_to_use
        self.size_embs = size_embs
        self.num_embs = num_embs
        
    def forward(self, inputs):
        matEmb = self.emb(inputs)
        twoWayInt = torch.bmm(matEmb, matEmb.permute(0,2,1))
        if self.mask_to_use is not None:
            twoWayInt[:, self.mask_to_use[:, 0], self.mask_to_use[:, 1]] = 0
        catEmb = matEmb.view(-1,self.num_embstimesSize)
        linPart = self.linearCat(catEmb)
        twoWayPart = self.linearTwoWay(twoWayInt.view(-1, self.num_embs *self.num_embs ))
        return linPart + twoWayPart

## This function restuns a vector [size_embs * num_embs,2] and is made to be used with the function
## CrossEntropyLoss() in Pytorch
class FwFM_ForCELoss(nn.Module):
    def __init__(self, size_vocab, size_embs, num_embs, mask_to_use, scale_grad_by_freq= False):
        super(FwFM_ForCELoss, self).__init__()
        self.emb = 0
        if torch.cuda.is_available():
            self.emb = nn.Embedding(size_vocab, size_embs, max_norm = 1, \
                scale_grad_by_freq = scale_grad_by_freq).cuda()
        else:
            self.emb = nn.Embedding(size_vocab, size_embs, max_norm = 1, \
                scale_grad_by_freq = scale_grad_by_freq)
        self.num_embstimesSize =  size_embs * num_embs
        self.linearCat = nn.Linear(self.num_embstimesSize, 2)
        self.linearTwoWay = nn.Linear(num_embs * num_embs, 2)
        self.mask_to_use = mask_to_use
        self.size_embs = size_embs
        self.num_embs = num_embs
        
    def forward(self, inputs):
        matEmb = self.emb(inputs)
        twoWayInt = torch.bmm(matEmb, matEmb.permute(0,2,1))
        if self.mask_to_use is not None:
            twoWayInt[:, self.mask_to_use[:, 0], self.mask_to_use[:, 1]] = 0
        catEmb = matEmb.view(-1,self.num_embstimesSize)
        linPart = self.linearCat(catEmb)
        twoWayPart = self.linearTwoWay(twoWayInt.view(-1, self.num_embs *self.num_embs ))
        return linPart + twoWayPart