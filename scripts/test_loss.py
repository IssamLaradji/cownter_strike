import sys, os
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np
from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src import datasets
# from src import optimizers 
import exp_configs
import torchvision

cudnn.benchmark = True

from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_img as hi
from haven import haven_results as hr
from haven import haven_chk as hc
# from src import looc_utils as lu
from PIL import Image

name2path = {'cityscapes':'/mnt/datasets/cityscapes',
             'pascal':'/mnt/datasets',
             'mri':'/mnt/private/datasets',
             'cows_json':'/mnt/cownter_strike'}

import pprint
import pandas as pd

if __name__ == "__main__":
    savedir_base = '/mnt/results/active_learning'
    for exp_group in [
                      'cows_counting',
                      ]:
        exp_list = exp_configs.EXP_GROUPS[exp_group]
        for exp_dict in exp_list:
            dataset_name = exp_dict['dataset']['name']
            loss_function = exp_dict['model']['loss']
            print('%s - %s' % (dataset_name,  loss_function))

            train_set = datasets.get_dataset(dataset_dict={'name':dataset_name},
                datadir=name2path[dataset_name], split="train",exp_dict=exp_dict)
            # for i in range(len(train_set)):
            #     batch = ut.collate_fn([train_set[12]])
            #     hu.save_image(fname='tmp.png', img=batch['images'], 
            #                   points=batch['points'])
            # original = hu.denormalize(b['images'], mode='rgb')
            batch = ut.collate_fn([train_set[12]])
            model = models.get_model(model_dict=exp_dict['model'],
                                            exp_dict=exp_dict,
                                            train_set=train_set).cuda()
            
            for i in range(100):
                val_meter = models.metrics.CountMeter(split='train')
                loss = model.train_on_batch(batch)
                gt_points = batch["points"]
                pred_points = model.predict_on_batch(batch, method='points')
                
                val_meter.update(pred_points=pred_points, gt_points=gt_points)
                
                print(i, 'loss:', loss['train_loss'], 'score:', val_meter.get_avg_score()['train_score'])
                
            model.vis_on_batch(batch, savedir_image='tmp.png')
            

