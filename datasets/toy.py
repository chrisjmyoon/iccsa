
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import pdb

from numpy.random import choice

"""
An implementation of a dataset to dynamically sample a pair of datapoints
    * (x1, y1) is sampled from p_df
    * (x2, y2) is sampled from s_dfs where y1 == y2 with P(p_yc)

Arguments
    * data - aggregate of all images generated
    * config - the config object from .hocon
    * p_df - a pandas dataframe containing a list of (index to data, label, domain)
    * s_dfs - a list of dataframes containing a list of (index to data, label, domain)
    * transform - a pytorch transform to apply to the data
    * p_yc - the probability where y1 == y2
"""
class ToyDataset(Dataset):
    def __init__(self, data, config, p_df, s_dfs, transform=None, p_yc=0.25):
        self.data = data
        self.n_sdomains = len(s_dfs)
        self.config = config
        self.p_df = p_df # Primary domain
        self.s_dfs = s_dfs # Seconday domains
        self.p_yc = p_yc
        self.transform = transform
        self.domain_p = self._get_domain_p()
    
    def __len__(self):
        return len(self.p_df)
    
    # Returns probability of sampling a domain from s_dfs based on the length of each domain
    def _get_domain_p(self):
        domain_p = [0]*self.n_sdomains
        n = 0.0
        for dom in self.s_dfs:
            n += len(dom)
        for i in range(self.n_sdomains):
            domain_p[i] = len(self.s_dfs[i])/n
        return domain_p

    # Sample x,y,d from secondary domains
    # if cl does not exist, sample from primary domain
    def _sample(self, domain, cl=None):
        if cl is None:
            df = self.s_dfs[domain]
        else:
            # Restrict sampling to only matching class
            d_df = self.s_dfs[domain]
            df = d_df[d_df["dx_idx"] == cl]
            if len(df) == 0:
                # If class does not exist, sample from primary domain
                d_df = self.p_df
                df = d_df[d_df["dx_idx"] == cl]

        idx = choice(len(df), size=1)[0]

        row = df.iloc[idx]
        dom = row['dom']
        x = self.data[int(row['path'])]
        y = row['dx_idx']
        return x, y, dom

    def __getitem__(self, idx):
        yc = choice([0,1], 1, p=[1-self.p_yc, self.p_yc])        
        # Sample a secondary domain
        domains = choice(self.n_sdomains, size=1, p=self.domain_p, replace=False) 
       
        # Get (x1, y1)
        d1 = self.p_df['dom'][idx]
        x1 = self.data[int(self.p_df['path'][idx])]
        y1 = int(self.p_df['dx_idx'][idx])

        cl = None
        if yc[0] == 1:
            cl = y1
    
        # Sample (x2, y2)
        x2, y2, d2 = self._sample(domains[0], cl=cl)
        
        y1 = torch.tensor(int(y1))
        y2 = torch.tensor(int(y2))
        yc = torch.tensor(int(yc[0]))

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1, x2, y1, y2, yc, d1, d2