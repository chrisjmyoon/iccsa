import pdb
import torch
import torch.nn as nn
import torchvision.models as torchmodels
import os
import numpy as np

from . import base
import graphs
import utils

"""
ToyAgent to demonstrate pipeline using the dynamic sampling and weighted CCSA Loss
"""
class ToyAgent(base.BaseAgent):
    def __init__(self, config, loaders, k):
        super().__init__(config, loaders, k) 
        
    # Gets  model from PyTorch database
    def _get_model(self):
        model = torchmodels.__dict__[self.config.arch](pretrained=self.config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, self.config.n_classes, bias=True)
        model.to(self.device)
        self.h = torch.nn.Sequential(*(list(model.children())[:-1])).to(self.device)
        self.classifier = model.fc.to(self.device)
        return model
   
    def _ml_logic(self, X, train=True):   
        loss_dict = dict()

        x1 = X[0].to(self.device) # data
        x2 = X[1].to(self.device)
        y1 = X[2].to(self.device) # labels
        y2 = X[3].to(self.device) 
        yc = X[4].to(self.device) # match?
        d1 = X[5] # domains
        d2 = X[6]

        
        # pass through model
        z1 = self.h(x1).squeeze()
        z2 = self.h(x2).squeeze()
        y1_pred = self.classifier(z1)
        y2_pred = self.classifier(z2)
        
        # Get criterion weighted based on observed data distributions
        self._update_counts(self.count1, y1)
        self._update_counts(self.count2, y2)
        self._update_pair_counts(self.pair_counts, y1, y2)
        self.criterion1 = nn.CrossEntropyLoss(weight=self._get_cb(self.count1)).cuda()
        self.criterion2 = nn.CrossEntropyLoss(weight=self._get_cb(self.count2)).cuda()
        ### Classification Loss
        loss_c1 = self.criterion1(y1_pred, y1)
        loss_c2 = self.criterion2(y2_pred, y2)
        loss_dict["loss_c1"] = loss_c1
        loss_dict["loss_c2"] = loss_c2

        ### Alignment Loss
        pair_cb = self._get_cb(self.pair_counts)
        loss_ca = graphs.loss.weighted_ca_loss(z1, z2, yc, y1, y2, pair_cb, margin=self.margin)
        loss_dict["loss_ca"] = loss_ca

        ### Combined Loss
        loss = (1 - self.alpha)*(loss_c1 + loss_c2) + self.alpha*loss_ca
        loss_dict["loss"] = loss
        return loss_dict
    
    def _val_ml_logic(self, X):      
        ml_results = dict()
        x = X[0].to(self.device)
        y = X[1].to(self.device)

        # Get predictions
        output = self.model(x)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.squeeze()

        ml_results["y"] = y.int().tolist()
        ml_results["pred"] = pred.int().tolist()
        return ml_results