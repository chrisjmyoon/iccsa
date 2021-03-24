import os
import pickle
import torch
import shutil
import pdb
from pathlib import Path

def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print("Created directory: {}".format(dir_path))
    else:
        print("Dir {} exists!".format(dir_path))


def save_pkl(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def save_torch_dict(fp, state):
    fp_dir = str(Path(fp).parent)
    if not os.path.isdir(fp_dir):
        os.makedirs(fp_dir)
    torch.save(state, fp)

def save_state(fp_dir, state, is_best=False, is_save_best=True):
    if not os.path.isdir(fp_dir):
        os.makedirs(fp_dir)
        
    checkpoint_path = os.path.join(fp_dir, "checkpoint.pth.tar")
    torch.save(state, checkpoint_path)
    if is_save_best:
        save_best(fp_dir, is_best)

def save_best(fp_dir, is_best, best_type="loss"):
    best_path = os.path.join(fp_dir, "best_{}_checkpoint.pth.tar".format(best_type))
    checkpoint_path = os.path.join(fp_dir, "checkpoint.pth.tar")
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
