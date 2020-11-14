import sys, os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
from haven import haven_img as hi
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

from src import models
from src import datasets
from src import utils as ut

import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src import utils as ut
cudnn.benchmark = True

if __name__ == "__main__":
    fname = '.tmp/val_pseudo.pkl'
    savedir_base = '/mnt/public/results/toolkit/cownter_strike'
    datadir = '/mnt/cownter_strike'

    hash_id = '21a289ad0c128aebfb557a8fe68a5fee'
    exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))

    train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                            split='train',
                                            datadir=datadir,
                                            exp_dict=exp_dict,
                                            dataset_size=exp_dict['dataset_size'])

    # Model
    # ==================
    model = models.get_model(model_dict=exp_dict['model'],
                            exp_dict=exp_dict,
                            train_set=train_set).cuda()

    model_path = os.path.join(savedir_base, hash_id, 'model_best.pth')

    # load best model
    model.load_state_dict(hu.torch_load(model_path))
    ind_list = np.where(train_set.labels)[0]
    for ind in ind_list[:100]:
        batch = ut.collate_fn([train_set[ind]])
        image=  hu.denormalize(batch['images'], mode='rgb')
        logits = model.model_base.forward(batch['images'].cuda())
        mask = hi.gray2cmap(logits.squeeze()/logits.max())
        hu.save_image('.tmp/density_%d.png' % ind, image.squeeze() * 0.5 + 0.5 * mask)
    print()