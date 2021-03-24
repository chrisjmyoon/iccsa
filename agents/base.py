import pdb
import shutil
import os
import torch
import utils
from utils.viz import *

"""
A base interface for an agent responsible for the training logic

Arguments
    * config - the config object from .hocon
    * loaders - a dict of PyTorch dataloaders
    * k - the fold index
"""
class BaseAgent:
    def __init__(self, config, loaders, k):
        self.config = config
        self.k = k

        # sets the artifact directory to store artifacts such as model
        self.artifact_dir = os.path.join(self.config.save_path, 
            str(self.config.rand_seed),
            self.config.exp_id,
            str(self.k))
        

        # sets device
        if self.config.gpu == "cpu" or not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        # sets loaders from dict()
        self._set_loaders(loaders)

        # initialize model
        self.model = self._get_model()

        # initialize meters and plotters to keep track of training process
        self.meters = dict()
        self.plotters = dict()

        # initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )        
        if self.config.lr_step:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.lr_step, gamma=0.1)
        
        # initialize class counts
        self._load_class_balance()

        # declare ccsa hyperparams
        self.margin = 50
        self.alpha = 0.1

        # initialize state
        self.epoch = 0
        self.best_loss = float('Inf')        
        self.best_wacc = 0

        # initialize visdom loggers
        if self.config.visdom:
            SingleVisdom.reset_window()

        # load from exising state
        if self.config.resume:
            self.load_checkpoint(file_path=self.config.resume)
        if self.config.retrain:
            self.load_model(file_path=self.config.retrain)
    
    """ 
    Load counts from file if exists, otherwise, empirically get counts
    """
    def _load_class_balance(self):
        cb_file_name = "{}_class_balance".format(self.config.exp_id)
        cb_dir = os.path.join(self.config.save_path, str(self.config.rand_seed), self.config.exp_id, str(self.k), "cb")
        cb_path = os.path.join(cb_dir, cb_file_name)
        
        if not os.path.isfile(cb_path):
            print("cb does not exist. Warmstarting...")
            n_classes = self.config.n_classes
            self.count1 = np.zeros(n_classes)
            self.count2 = np.zeros(n_classes)
            self.pair_counts = np.zeros((n_classes,n_classes))
            self._warmstart_counts()
            print("Saved cb: {}".format(cb_path))
            self._save_cb(self.count1, self.count2, self.pair_counts)
        else:
            print("Loading cb: {}".format(cb_path))
            self.count1, self.count2, self.pair_counts = utils.fs.load_pkl(cb_path)
    """
    Go through n_warmstart epochs to get empirical distribution of labels
    """
    def _warmstart_counts(self):
        n_warmstart = 10
        for w in range(n_warmstart):
            print("Warmstart: {}/{}".format(w, n_warmstart))
            for i, X in enumerate(self.tr_loader):  
                y1 = X[2] # labels
                y2 = X[3]          

                self._update_counts(self.count1, y1)
                self._update_counts(self.count2, y2)
                self._update_pair_counts(self.pair_counts, y1, y2)
    
    # Helper to update emprical class distribution
    def _update_counts(self, counts, sampled):
        for i in sampled.tolist():
            counts[i] += 1

    # Helper to update emprical class distribution
    def _update_pair_counts(self, counts, y1, y2):
        l_y1 = y1.tolist()
        l_y2 = y2.tolist()

        for i in range(len(l_y1)):
            counts[y1[i]][y2[i]] += 1

    """
    Saves count1, count2, pair_counts
    """
    def _save_cb(self, count1, count2, pair_counts):
        cb_file_name = "{}_class_balance".format(self.config.exp_id)
        cb_dir = os.path.join(self.config.save_path, str(self.config.rand_seed), self.config.exp_id, str(self.k), "cb")
        cb_path = os.path.join(cb_dir, cb_file_name)
        utils.fs.create_dir(cb_dir)
        cb_obj = [count1, count2, pair_counts]
        utils.fs.save_pkl(cb_obj, cb_path)
    
    """
    Computes the weight based on class distribution
    """
    def _get_cb(self, counts):
        # handle when count is 0
        # if count is zero, use the boost of the smallest nonzero class count
        c_copy = counts.copy()
        min_count = c_copy[c_copy != 0].min()
        c_copy[c_copy == 0] = min_count

        # compute class boost
        cb = np.sum(counts) / c_copy

        # Normalize to 0-1
        n_cb = cb.astype('float') / np.max(cb) 
        n_cb = torch.tensor(n_cb).float()
        return n_cb

    def _set_loaders(self, loaders):
        self.tr_loader = loaders["tr_loader"]
        self.val_loaders = loaders["val_loaders"]
        self.len_tr_loader = len(self.tr_loader)
        
    def load_checkpoint(self, file_path=None):
        if os.path.isfile(file_path):
            state = torch.load(file_path)
            print("Loaded {}".format(file_path))
        else:
            print("{} does not exist.".format(file_path))
            raise FileNotFoundError
        self.model = state['model']
        self.optimizer = state['optimizer']
        self.epoch = state['epoch']
        # Robustly loading state["meters"] even if state doesn't contain newer keys
        for meter_name, meter in state["meters"].items():
            self.meters[meter_name] = meter
       
        self.best_loss = state['best_loss']
        self.best_wacc = state['best_wacc']
        visdom_state = state["visdom_state"]
        self._load_visdom_state(visdom_state)
        print("Loaded checkpoint to ADNI Agent")
    
    def load_model(self, file_path=None):
        state = super().load_checkpoint(file_path)
        self.model = state['model']

    # Saves model state to checkpoint_path
    def save_checkpoint(self, epoch, is_best=False):
        visdom_state = self._get_visdom_state()
        state = {
            'epoch': epoch,
            'model': self.model,
            'optimizer': self.optimizer,
            'meters': self.meters,
            'best_loss': self.best_loss,
            'best_wacc': self.best_wacc,
            'visdom_state': visdom_state,
        }

        utils.fs.create_dir(self.artifact_dir)

        checkpoint_path = os.path.join(self.artifact_dir, "checkpoint.pth.tar")
        best_path = os.path.join(os.path.split(checkpoint_path)[0], "best_checkpoint.pth.tar")
        torch.save(state, checkpoint_path)

        if is_best:
            shutil.copyfile(checkpoint_path, best_path)
        
        epoch = state["epoch"]
        epoch_path = os.path.join(os.path.split(checkpoint_path)[0], "{}_checkpoint.pth.tar".format(epoch))
        # if epoch % 50 == 0:
        #     shutil.copyfile(checkpoint_path, epoch_path)
        print("Saved checkpoint")
 
    
    def _print(self, i, max_length, meters, is_train=True):        
        print(  "Epoch: [{0}][{1}/{2}]\t  Loss: {loss_val:.4f}\t".format(
                            self.epoch, i, max_length,
                            loss_val= meters["loss"].avg))

    def _plot(self, meters, is_train=True):
        for loss_name, loss_meter in meters.items():
            if loss_name not in self.plotters:
                self.plotters[loss_name] = utils.viz.LossPlotter(loss_name, title="{} Loss".format(loss_name), len_loader=self.len_tr_loader, epoch=self.epoch)
            
            if is_train:
                self.plotters[loss_name].plot_train(loss_meter.avg)
            else:
                self.plotters[loss_name].plot_val(loss_meter.avg)

    def _get_batch_size(self, X):
        return X[0].size(0)           

    def _update_meters(self, X, ml_results, meters):
        batch_size = self._get_batch_size(X)
        for loss_name, loss_val in ml_results.items():
            if loss_name not in meters:
                meters[loss_name] = utils.meters.AverageMeter()
            meters[loss_name].update(loss_val.item(), batch_size)   

    def _get_visdom_state(self):
        visdom_state = {plotter_name: plotter.export_state() for plotter_name, plotter in self.plotters.items()}
        return visdom_state

    def _load_visdom_state(self, visdom_state):
        for plotter_name, plotter_state in visdom_state.items():
            if plotter_name not in self.plotters:
                self.plotters[plotter_name] = utils.viz.LossPlotter(plotter_name, title="{} Loss".format(plotter_name), len_loader=self.len_tr_loader, epoch=self.epoch)
            self.plotters[plotter_name].load_from_state(plotter_state)

    def train(self):
        for epoch in range(self.epoch, self.config.epochs):
            self._train_epoch()
            wacc = self.validate()

            is_best = wacc > self.best_wacc
            if is_best:
                self.best_wacc = wacc
            if self.config.lr_step:
                self.scheduler.step()
            self.save_checkpoint(epoch + 1, is_best=is_best)
            self.epoch += 1

    def _train_epoch(self):      
        self.model.train()

        for meter in self.meters.values():
            meter.reset()
            
        self.optimizer.zero_grad()
        for i, X in enumerate(self.tr_loader):
            # Update plotter
            for plotter_name, plotter in self.plotters.items():
                plotter.increment_train_epoch()

            if self._get_batch_size(X) < self.config.batch_size:
                continue

            if self.config.test_mode and i == self.config.test_mode:
                print("Stopping training early because test_mode is True")
                break    

            ml_results = self._ml_logic(X, train=True)
            loss = ml_results["loss"]
            loss.backward()
            
            self.optimizer.step() 
            self.optimizer.zero_grad()

            # Update losses
            self._update_meters(X, ml_results, self.meters)

            # print every config.print_freq
            if i % self.config.print_freq == 0:
                self._print(i, self.len_tr_loader, self.meters, is_train=True)

                # Plot to visdom
                if self.config.visdom:
                    self._plot(self.meters, is_train=True)

                # Reset meters to prevent skewed losses
                for meter in self.meters.values():
                    meter.save_reset() 
                # loss_meter.save_reset()
        print("Finished training epoch: {}".format(self.epoch))  


    def validate(self):
        print("starting to validate")
        wacc_meter = utils.meters.AverageMeter()
        for val_number, val_loader in enumerate(self.val_loaders):
            val_plotter_name = "val_plotter_{}".format(val_number)
            if val_plotter_name not in self.plotters:
                self.plotters[val_plotter_name] = utils.viz.LossPlotter(val_plotter_name, 
                    title=val_plotter_name, 
                    len_loader=1, 
                    epoch=self.epoch)
            wacc = self._validate(val_loader, val_plotter_name)
            wacc_meter.update(wacc)
        print("Overall validation wacc: {}".format(wacc_meter.avg))
        return wacc_meter.avg

    # Validates using paired images
    def _validate(self, val_loader, val_plotter_name):     
        val_meters = dict()
        
        y_preds = []
        y_trues = []

        self.model.eval()
        with torch.no_grad():
            for i, X in enumerate(val_loader):
                # if self.config.test_mode and i == self.config.test_mode:
                #     print("Stopping validation early because test_mode is True")
                #     break

                ml_results = self._val_ml_logic(X)              
                
                y_preds.extend(ml_results["pred"])
                y_trues.extend(ml_results["y"])
                
                # print every config.print_freq
                if i % self.config.print_freq == 0:
                    print(  "Validating {}: [{}][{}/{}]".format(
                            val_plotter_name, self.epoch, i, len(val_loader)))
        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)                   

        # Compute balanced accuracy
        unique_classes = np.unique(y_trues)
        accuracies = dict()
        for cl in unique_classes:            
            cl_indices = (y_trues == cl)
            n_cl = cl_indices.sum()
            n_correct = (y_preds[cl_indices] == y_trues[cl_indices]).sum()
            accuracies[cl] = n_correct / n_cl

        np_accuracies = np.array(list(accuracies.values()))
        print("{}: {}".format(val_plotter_name, accuracies))
        wacc = np_accuracies.mean()

        # Plot to visdom
        if self.config.visdom:
            self.plotters[val_plotter_name].increment_val_epoch()
            self.plotters[val_plotter_name].plot_val(wacc)
        return wacc