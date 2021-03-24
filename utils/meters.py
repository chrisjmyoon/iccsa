import numpy as np
import pdb 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.history = []

    # Stores average to history and resets
    def save_reset(self):
        self.history.append(self.avg)
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AggregateMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.history = []
        self.min = None
        self.max = None
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 0
        self.max = 0

    def update(self, val, n=1):
        self.history.append(val)
        # store avg
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # store min / max
        if self.min is None:
            self.min = val
        if self.max is None:
            self.max = val
        if val < self.min:
            self.min = val
        if val > self.max:
            self.max = val
    
    def update_list(self, listval):
        for val in listval:
            self.update(val, n=1)


class MinMaxMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()        
    
    def reset(self):
        self.min = None
        self.max = None

    def update(self, val, n=1):        
        # store min / max
        if self.min is None:
            self.min = val
        if self.max is None:
            self.max = val
            
        if val < self.min:
            self.min = val
        if val > self.max:
            self.max = val
    
    def update_list(self, listval):
        for val in listval:
            self.update(val)

class LastKMeter:
    def __init__(self, k=50):
        self.k = k
        self.reset()               
    
    def reset(self):
        self.history = np.zeros(self.k)
        self.count = 0
    
    def update(self, val):
        index = self.count % self.k
        self.history[index] = val
        self.count += 1        

    def update_list(self, list_val):
        for val in list_val:
            self.update(val)
    
    def get_avg(self):
        if self.count == 0:
            return 0
        elif self.count < self.k:
            # return self.history[:self.count].mean()
            return 0 # wait until a good mean is established
        else:
            return self.history.mean()
    
    def get_max(self):
        return self.history.max()
