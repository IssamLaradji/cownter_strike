import sys, os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

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

if __name__ == "__main__":
    fname = '.tmp/val_pseudo.pkl'
    savedir_base = '/mnt/public/results/toolkit/cownter_strike'
    datadir = '/mnt/cownter_strike'

    if os.path.exists(fname) and 0:
        score_list = hu.load_pkl(fname)
    else:
        hash_list = []
       
        # density
        hash_list += ['21a289ad0c128aebfb557a8fe68a5fee']

        # lcfcn_nopretrain
        hash_list += ['7fe3bf4ff0078177506ee63b32cd9a66']

        # lcfcn
        hash_list += ['ec8d2f5bba68512cff946444b0d19780']

        score_list = {}
        for hash_id in  hash_list:
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

            
            for s_e_count in [(0,100000)]:
                test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                                split='test',
                                                datadir=datadir,
                                                exp_dict=exp_dict,
                                                dataset_size=exp_dict['dataset_size'])

                # get sub tests
                meta_list_new = [v for  v in test_set.meta_list
                                if  s_e_count[0] <= v['count']  and v['count'] <=  s_e_count[1]]
                print(len(meta_list_new), '/', len(test_set), '-', s_e_count)
                test_set.meta_list = meta_list_new
                test_loader = DataLoader(test_set,
                                        batch_size=1,
                                        collate_fn=ut.collate_fn,
                                        num_workers=0)

                test_dict = model.val_on_loader(test_loader, keep_counts=True)
                print('results for hash: %s' % hash_id)
                pprint.pprint(test_dict)
                score_list["%s_%s" % (hash_id, s_e_count)] = test_dict
        hu.save_pkl(fname, score_list)

    df = pd.DataFrame(score_list)

    count_list_dict = {}
    for k, v in score_list.items():
        count_list_dict[k.split('_')[0]] = v['count_list']
    
    hu.save_json('.tmp/count_list.json',count_list_dict )
    
    count_results = 0
    # Count
    if not count_results:
        for s_e_count in [(0,100000)]:
            subset = [c for c in df.columns if str(s_e_count) in c]
            df_list = []
            for s in subset:
                hash_id = s.split('_')[0]
                exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))
                loss = exp_dict['model']['loss']
                score_dict = score_list["%s_%s" % (hash_id, s_e_count)]
                df_list += [{'loss':loss, 
                            'accuracy %s' % str(s_e_count):score_dict['test_acc'], 
                            'fscsore %s' % str(s_e_count):score_dict['test_fscore'],
                            'always median acc %s' % str(s_e_count):1 - score_dict['test_prec_always_1'],
                            'always median fscore %s' % str(s_e_count):0}]
            
            print(pd.DataFrame(df_list))
    else:
        # Count
        for s_e_count in [(1,10), (11,100), (101,10000), (0,0)]:
            subset = [c for c in df.columns if str(s_e_count) in c]
            df_list = []
            for s in subset:
                hash_id = s.split('_')[0]
                exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))
                loss = exp_dict['model']['loss']
                score_dict = score_list["%s_%s" % (hash_id, s_e_count)]
                df_list += [{'loss':loss, 
                            'mae %s' % str(s_e_count):-score_dict['test_score'], 
                            'game %s' % str(s_e_count):-score_dict['test_game'],
                            'magame %s' % str(s_e_count):-score_dict['test_magame'],
                            'mape %s' % str(s_e_count):-score_dict['test_mape'],
                            'always median %s' % str(s_e_count):-score_dict['test_always_0']
                }]
            
            print(pd.DataFrame(df_list))
            print()

    print()