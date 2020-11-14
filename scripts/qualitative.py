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

cudnn.benchmark = True

name_dict = {'lcfcn': 'LCFCN', 'density':'CSRNet'}
if __name__ == "__main__":
    savedir_base = '/mnt/public/results/toolkit/cownter_strike'
    # hash_list = []
    datadir = '/mnt/cownter_strike'
    # on localiization
    hash_list = [#density
                 'c9e53fe0212bf5da731436b9de3ac582',
                #  # LCFCN no pretrain
                #  'b2b12bd18afcef4d4e5444b4298b381b',
                 #LCFCN
                 '91e06f8aa948d0c0d6abc1c642ed8344']

    main_hash = 'c9e53fe0212bf5da731436b9de3ac582'
    exp_dict = hu.load_json(os.path.join(savedir_base, main_hash, 'exp_dict.json'))
    
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                            split='test',
                                            datadir=datadir,
                                            exp_dict=exp_dict,
                                            dataset_size=exp_dict['dataset_size'])
    test_loader = DataLoader(test_set,
                                    # sampler=val_sampler,
                                    batch_size=1,
                                    collate_fn=ut.collate_fn,
                                    num_workers=0)

    for i, batch in enumerate(test_loader):
        points = (batch['points'].squeeze() == 1).numpy()
        if points.sum() == 0:
            continue
        savedir_image = os.path.join('.tmp/qualitative/%d.png' % (i))
        img = hu.denormalize(batch['images'], mode='rgb')
        img_org = np.array(hu.save_image(savedir_image, img, points=batch['points'].numpy(), radius=2, return_image=True))
        img_org2 = np.array(hu.save_image(savedir_image, img, return_image=True))
        img_org = hi.text_on_image( 'True Count: %d' % int(batch['points'].sum()), 
           img_org)
        img_org2 = hi.text_on_image( 'Original Image', 
           img_org2)
        img_list = [img_org2, img_org]
        with torch.no_grad():
            for hash_id in hash_list:
                score_path = os.path.join(savedir_base, hash_id, 'score_list_best.pkl')
                score_list = hu.load_pkl(score_path)
                
                exp_dict = hu.load_json(os.path.join(savedir_base, hash_id, 'exp_dict.json'))
                # print(i, exp_dict['model']['loss'], exp_dict['model'].get('with_affinity'), 'score:', score_list[-1]['test_class1'])
                
                model = models.get_model(model_dict=exp_dict['model'],
                                            exp_dict=exp_dict,
                                            train_set=test_set).cuda()

                model_path = os.path.join(savedir_base, hash_id, 'model_best.pth')
                model.load_state_dict(hu.torch_load(model_path))
                if exp_dict['model']['loss'] == 'density':
                    image=  hu.denormalize(batch['images'], mode='rgb')
                    logits = model.model_base.forward(batch['images'].cuda())
                    mask = hi.gray2cmap(logits.squeeze()/logits.max())
                    img_pred = image.squeeze() * 0.5 + 0.5 * mask
                    img_pred = np.array(hu.save_image(savedir_image, img_pred, return_image=True))
                else:
                    mask_pred = model.predict_on_batch(batch)
                    img_pred = np.array(hu.save_image(savedir_image, img, mask=mask_pred,  return_image=True))
                img_pred = hi.text_on_image( '%s Count: %d' % 
                        (name_dict[exp_dict['model']['loss']], 
                            int(model.predict_on_batch(batch, method='counts'))), img_pred)
                img_list += [img_pred]

        img_cat = np.concatenate(img_list, axis=1)
        hu.save_image(savedir_image, img_cat)
        print(i, '/', len(test_loader))

          