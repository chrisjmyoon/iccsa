import pdb
from pyhocon import ConfigFactory
import os

"""
Loads configurations set in a .hocon file to a Python object
Refer to the hocon file for description of variables
"""
class BaseConfig:
    def __init__(self, conf_path):
        super().__init__()
        self.conf = ConfigFactory.parse_file(conf_path)

        ### General
        self.exp_id = self.conf['exp_id']
        self.dataset = self.conf['dataset']
        self.primary_dataset = self.conf['primary_dataset']
        self.gpu = self.conf['gpu'] 
        self.print_freq = self.conf['print_freq']
        self.num_workers = self.conf['num_workers']  
        self.rand_seed = self.conf['rand_seed']
        self.visdom = self.conf['visdom']
        self.test_mode = self.conf['test_mode']
        
        ### Optimization
        self.batch_size = self.conf['batch_size']  
        self.weight_decay = self.conf['weight_decay'] 
        self.lr = self.conf['lr']   
        self.epochs = self.conf['epochs']   
        self.lr_step = self.conf['lr_step']

        ### Model
        self.arch = self.conf['arch']
        self.n_classes = self.conf['n_classes']
        self.pretrained = self.conf['pretrained']  

        ### File Path
        self.save_path = os.path.expanduser(self.conf['save_path'])
        self.data_path = os.path.expanduser(self.conf['data_path'])
        self.split_path = os.path.expanduser(self.conf['split_path'])
        self.retrain = None if len(self.conf['retrain']) == 0 else os.path.expanduser(self.conf['retrain'])
        self.resume = None if len(self.conf['resume']) == 0 else os.path.expanduser(self.conf['resume'])

        ### Experiment
        self.num_secondary = self.conf['num_secondary']
        self.removed_dataset = self.conf['removed_dataset']
        self.mode = self.conf['mode']
