from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import time
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('.'))
from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True


def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0):
    # bookkeepting stuff
    # ==================
    pprint.pprint(exp_dict)
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    print(savedir)

    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # ==================
    # train set
    val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])


    # val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
                            # sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)

    # Model
    # ==================
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=val_set).cuda()

    model_path = os.path.join(savedir, "model.pth")
    model_path = '/mnt/public/results/toolkit/cownter_strike/5389cdcb28af61f9515bbf5bd47154f5/model.pth'

    # resume experiment
    model.load_state_dict(hu.torch_load(model_path))

    # model.opt = optimizers.get_optim(exp_dict['opt'], model)

    # Train & Val
    # ==================
    # train_sampler = torch.utils.data.RandomSampler(
    #                             train_set, replacement=True, 
    #                             num_samples=len(val_set))

    model.vis_on_loader(val_loader, savedir_images=os.path.join(savedir, "images"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default=None)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    args = parser.parse_args()
    exp_dict = {
                     'batch_size': 1,
                        'num_channels':1,
                        'dataset': {'name':'cows'} ,
                        'dataset_size':{'train':'all', 'val':'all'},
                        # 'dataset_size':{'train':10, 'val':10},
                        'max_epoch': 500,
                        'optimizer':  "adam", 
                        'lr':  1e-5,
                        'model': {'name':'semseg', 'loss':'lcfcn',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':1}}
                        
    trainval(exp_dict=exp_dict,
            savedir_base=args.savedir_base,
            datadir=args.datadir,
            reset=args.reset,
            num_workers=args.num_workers)
        
