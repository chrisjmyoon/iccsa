import pdb
import os
import utils
import torchvision.transforms as transforms
import pandas as pd
import torch
import numpy as np 
import datasets

"""
Basic interface for experiments
"""
class BaseExperiment:
    def __init__(self, config, k):
        self.config = config
        self.k = k
        """ creates cv splits folder if not exists """
        # create dir
        cv_split_path = os.path.join(self.config.split_path, "cv", str(self.config.rand_seed), self.config.dataset)
        is_create_splits = not os.path.isdir(cv_split_path)
        # is_create_splits = True
        if is_create_splits:
            self.create_splits()
        else:
            print("{} exists".format(cv_split_path))


        artifact_dir = os.path.join(self.config.save_path, 
            str(self.config.rand_seed),
            self.config.exp_id,
            str(self.k))

        # ensures artifacts folder is created
        utils.fs.create_dir(artifact_dir)
        self.loaders = self.get_loaders()
        self.agent = self.get_agent()

    def get_loaders(self):
        raise NotImplementedError

    def get_agents(self):
        raise NotImplementedError

    def create_splits(self):
        raise NotImplementedError

    """
    Run agent with different endpoints
    """
    def run(self):
        if self.config.mode == 'train':
            self.agent.train()     
        else:
            raise Exception("conf['mode'] invalid!")
        print("Finished running experiment")

