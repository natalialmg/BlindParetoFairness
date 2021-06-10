import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import sys
sys.path.append("../")
from .utils import to_np
import numpy as np

class losses(nn.Module):
    def __init__(self,type_loss = 'L2',regression = False, reduction = 'sum' ):
        super(losses, self).__init__()
        self.reduction=reduction
        self.type_loss = type_loss
        self.regression = regression # if false (default) we assume classification
    def forward(self, inputs, targets):

        if self.type_loss == 'L1':
            if not self.regression:
                inputs = torch.nn.Softmax(dim=-1)(inputs) #last dim are the classes
            ret = torch.abs(inputs - targets)

        elif self.type_loss == 'CE':
            if not self.regression:
                inputs = torch.nn.LogSoftmax(dim=-1)(inputs)
            ret = -1*(inputs*targets)

        else: #default is L2 (MSE or Brier score )
            if not self.regression:
                inputs = torch.nn.Softmax(dim=-1)(inputs)
            ret = (inputs - targets) ** 2

        if self.reduction != 'none':
            ret = torch.mean(ret,-1) if self.reduction == 'mean' else torch.sum(ret,-1)

        return ret

class metrics(nn.Module):
    def __init__(self,type_loss = 'acc'):
        super(metrics, self).__init__()
        self.type_loss = type_loss

    def forward(self, inputs, targets):

        if self.type_loss == 'R2':
            inputs = to_np(inputs)  # last dim are the classes
            targets = to_np(targets)
            ret = np.sum((inputs-targets)**2) / np.sum((inputs - np.mean(targets))**2)

        if self.type_loss == 'acc':
            inputs = torch.nn.Softmax(dim=-1)(inputs)  # last dim are the classes
            input_argmax = np.argmax(to_np(inputs),-1)
            ret = to_np(targets)[np.arange(input_argmax.shape[0]),input_argmax]

        if self.type_loss == 'err':
            inputs = torch.nn.Softmax(dim=-1)(inputs)  # last dim are the classes
            input_argmax = np.argmax(to_np(inputs),-1)
            ret = 1-to_np(targets)[np.arange(input_argmax.shape[0]),input_argmax]

        if self.type_loss == 'softacc':
            inputs = torch.nn.Softmax(dim=-1)(inputs)  # last dim are the classes
            ret = np.sum(to_np(inputs)*to_np(targets),-1)

        if self.type_loss == 'softerr':
            inputs = torch.nn.Softmax(dim=-1)(inputs)  # last dim are the classes
            ret = 1-np.sum(to_np(inputs)*to_np(targets),-1)

        if self.type_loss == 'auc':
            inputs = torch.nn.Softmax(dim=-1)(inputs)  # last dim are the classes
            y_true = to_np(targets)
            y_pred = to_np(inputs)
            # print(roc_auc_score(y_true, y_pred))
            # return torch.from_numpy(np.array([roc_auc_score(y_true, y_pred)]))
            return np.array([roc_auc_score(y_true, y_pred)])

        return ret