import numpy as np
import torch
import imageio
import os
import copy
import pydicom
from PIL import Image
from haven import haven_utils as hu
# from src import proposals
from src import datasets
from skimage.io import imread
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as FT
import numpy as np
import torch
from skimage import morphology
from skimage.io import imread
import torchvision.transforms.functional as FT
from skimage.transform import rescale
import torchvision
from torchvision.transforms import transforms
import pylab as plt
from skimage.color import label2rgb
from skimage.segmentation import slic
from haven import haven_utils as hu
from haven import haven_img as hi
from src import utils as ut
# from repos.aranxta_code.extract_cost import CsObject
# from repos.selectivesearch.selectivesearch import selective_search
# from src import region_methods
# import pycocotools.mask as mask_util
from skimage.segmentation import mark_boundaries
import pandas as pd 
import glob
import urllib, json


class Cows:
    def __init__(self, datadir, split, exp_dict=None):
        self.split = split
        self.exp_dict = exp_dict
        self.meta_list = []
        self.split_number = exp_dict["dataset"]["%s_split_number" %split]
        self.stratification = exp_dict["dataset"]["stratification"]
        meta = hu.load_json(os.path.join(datadir, 
                                         "splits",
                                         self.stratification,
                                         "%s_%d.json" %(split, self.split_number)))
        self.labels = []
        self.meta_list = []
        for path, attributes in meta.items():
            label_dict = attributes["points"]
            if len(label_dict) == 0:
                point_list = []
            else:
                point_list = list(list(label_dict.values())[0].values())
            fname = os.path.join(datadir, path)
            meta_dict = {'fname':fname,
                            'point_list':point_list,
                            'count':len(point_list)}            
            self.labels.append(int(len(point_list) > 0))
            self.meta_list.append(meta_dict)

        self.labels = np.array(self.labels)
        print('Foreground Ratio: %.3f' % self.labels.mean())
        self.transforms = None
                                                                         
    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, index):
        # index = 0
        meta_dict = self.meta_list[index]
        image_pil = Image.open(meta_dict['fname'])
        if self.exp_dict['dataset'].get('transform') == 'resize':
            image_pil = torchvision.transforms.Resize((256, 256))(image_pil)

        w, h = image_pil.size
        image = torchvision.transforms.ToTensor()(image_pil)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        image = transforms.Normalize(mean=mean, std=std)(image)

        point_list = meta_dict['point_list']
  
        points = torch.zeros((h, w))
        for p in point_list:
            if 'y' not in p or p['y'] >= h or p['x'] >= w:
                continue
            points[int(p['y']), int(p['x'])] = 1
        

        batch = {'images':image,
                #  'ndvi':,
                 'labels':torch.as_tensor(len(point_list)>0).long(),
                 'points':points.long(),
                 'point_list':point_list,
                 'meta':{'index':index, 'name':meta_dict['fname'],
                         'id':index, 'split':self.split}}
        if 'fname_ndvi' in meta_dict:
            image_pil = Image.open(meta_dict['fname_ndvi'])
            w, h = image_pil.size
            image = torchvision.transforms.ToTensor()(image_pil)

            batch['ndvi'] = image[0,:,w//2:]

        return batch
