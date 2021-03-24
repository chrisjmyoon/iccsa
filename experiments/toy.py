import pdb
import os
import utils
import torchvision.transforms as transforms
import pandas as pd
import torch
import numpy as np 
import datasets
import agents
from . import base

"""
ToyExperiment contains code to generate a synthetic toy dataset
"""
class ToyExperiment(base.BaseExperiment):
    def __init__(self, config, k):
        super().__init__(config, k)

    """
    Generates a toy dataset of squares
    """
    def generate_data(self, distribution, dataset_index):
        # Returns a square with mean_intensity
        def make_square(square_len=32, mean_intensity=0):
            square = np.zeros((square_len, square_len, 3))
            square_1d = np.random.normal(
                loc=mean_intensity, 
                scale=1, 
                size=(square_len, square_len, 1))
            square = np.tile(square_1d, (1,1,3))
            return square  
        
        N = sum(distribution)
        data = np.zeros((N, 64, 64, 3)).astype('float32')
        labels = np.zeros((N))
        # Class determines intensity
        # Dataset determines location
        position_multiplier = 2
        intensity_multiplier = 5
        count = 0
        square_len = 32
        for cl, num_cl in enumerate(distribution):
            for i in range(num_cl):
                mean_intensity = np.random.normal(loc=1+cl*intensity_multiplier)
                x_pos = dataset_index*position_multiplier
                y_pos = dataset_index*position_multiplier
                data[count][x_pos:x_pos+square_len, y_pos:y_pos+square_len] = make_square(square_len, mean_intensity)
                
                labels[count] = cl
                count += 1
        return data, labels
      
    """
    Create a dataset and dataloader based on distribution of classes per domain
    """
    def create_dataloader(self, is_train=True):
        # Class Distributions
        dists = [
            [1000, 800, 600, 400, 200], 
            [0, 1000, 1000, 0, 0],
            [0, 0, 1000, 1000, 0],
            [0, 0, 0, 1000, 1000]
        ]
        N_per_dataset = np.array([sum(dist) for dist in dists])
        N_datasets = len(dists)

        # Generate 5 datasets based on distributions
        datas = []
        labels = []
        for i, dist in enumerate(dists):
            data, label = self.generate_data(dist, i)
            datas.append(data)
            labels.append(label)
        agg_data = np.concatenate(datas)

        # Create dataframes based on dataset
        # Each row contains [data, label, dataset]
        def create_df(start_idx, data, labels, dataset_index):
            df = pd.DataFrame()
            df["path"] = list(range(len(data))) # path/index to data in agg_data
            df["dx_idx"] = labels
            df["dom"] = dataset_index
            return df
        dfs = []
        for i in range(N_datasets):
            start_idx = N_per_dataset[0:i].sum()
            df = create_df(start_idx, datas[i], labels[i], i)
            dfs.append(df)
        # Create dataset and dataloader
        p_df = dfs[0]
        s_dfs = dfs[1:]
        data = agg_data
        tr = transforms.Compose([
            transforms.ToTensor()
        ])

        if is_train:
            dataset = datasets.toy.ToyDataset(data, self.config, p_df, s_dfs, transform=tr, p_yc=0.25)
            # Randomly seed each worker
            def worker_init_fn(worker_id):    
                np.random.seed(np.random.get_state()[1][0] + worker_id)
            
            loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=self.config.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=True)
            return loader
        else:
            loaders = []
            for i in range(N_datasets):
                dataset = datasets.simple.SimpleDataset(datas[i], labels[i], tr)
                loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=self.config.batch_size, 
                    shuffle=False, 
                    num_workers=self.config.num_workers,
                    pin_memory=True
                )
                loaders.append(loader)
            return loaders           


    def get_loaders(self):        
        tr_loader = self.create_dataloader(is_train=True)
        val_loaders = self.create_dataloader(is_train=False)
        loaders = dict(
            tr_loader = tr_loader,
            val_loaders = val_loaders
        )
        return loaders

    def get_agent(self):
        agent = agents.toy.ToyAgent(self.config, self.loaders, self.k)
        return agent

    def create_splits(self):
        pass