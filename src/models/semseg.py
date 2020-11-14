# -*- coding: utf-8 -*-

import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import infnet, fcn8_vgg16, unet_resnet, resnet_seam
from src import utils as ut
from src import models
from . import losses
from src.modules.lcfcn import lcfcn_loss
import sys
from kornia.geometry.transform import flips


try:
    import kornia
    from kornia.augmentation import RandomAffine
    from kornia.geometry.transform import flips
except:
    print('kornia not installed')
    
from scipy.ndimage.filters import gaussian_filter

from . import optimizers, metrics, networks
from src.modules import sstransforms as sst

class SemSeg(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict = exp_dict
        self.train_hashes = set()
        self.n_classes = self.exp_dict['model'].get('n_classes', 1)

        self.init_model()
        self.first_time = True
        self.epoch = 0

    def init_model(self):
        self.model_base = networks.get_network(self.exp_dict['model']['base'],
                                              n_classes=self.n_classes,
                                              exp_dict=self.exp_dict)
        self.cuda()
        self.opt = optimizers.get_optimizer(self.exp_dict['optimizer'], self.model_base, self.exp_dict)


    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict(),
                      'epoch':self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        if 'opt' not in state_dict:
            return
        self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict['epoch']

    def train_on_loader(self, train_loader):
        
        self.train()
        self.epoch += 1
        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()
    
        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_monitor.add(score_dict)
            msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            pbar.update(1)
            
        pbar.close()

        return train_monitor.get_avg_score()

    def train_on_batch(self, batch):
        # add to seen images
        # for m in batch['meta']:
        #     self.train_hashes.add(m['hash'])

        self.opt.zero_grad()

        images = batch["images"].cuda()
       

        
        # compute loss
        loss_name = self.exp_dict['model']['loss']
        if loss_name in 'cross_entropy':
            logits = self.model_base(images)
            # full supervision
            loss = losses.compute_cross_entropy(images, logits, masks=batch["masks"].cuda())
        
        elif loss_name in 'point_level':
            logits = self.model_base(images)
            # point supervision
            loss = losses.compute_point_level(images, logits, point_list=batch['point_list'])
        
        elif loss_name in ['lcfcn_ndvi']:
            images = torch.cat([images, batch["ndvi"][0].cuda()[None,None]], dim=1)
            logits = self.model_base(images)
            loss = lcfcn_loss.compute_loss(points=batch['points'], probs=logits.sigmoid())
    
        elif loss_name in ['lcfcn', 'lcfcn_nopretrain']:
            # implementation needed
            logits = self.model_base(images)
            loss = lcfcn_loss.compute_loss(points=batch['points'], probs=logits.sigmoid())

        elif loss_name in 'prm':
            counts = batch['points'].sum()

            logits = self.model_base(images)
            peak_map = get_peak_map(logits,
                                win_size=3,
                                counts=counts)
            act_avg = (logits * peak_map).sum((2,3)) / peak_map.sum((2,3))
            loss = F.mse_loss(act_avg.squeeze(), counts.float().squeeze().cuda())

        elif loss_name in 'cam':
            # implementation needed
            counts = batch['points'].sum()
            logits = self.model_base(images).mean()
            loss = F.binary_cross_entropy_with_logits(logits, (counts>0).float().squeeze().cuda(), reduction='mean')


        elif loss_name in 'prm_points':
            # implementation needed
            counts = batch['points'].sum()
            logits = self.model_base(images)
            peak_map = batch['points'].cuda()[None]
            logits_avg = (logits * peak_map).mean()
            loss = F.binary_cross_entropy_with_logits(logits_avg, (counts>0).float().squeeze().cuda(), reduction='mean')

        elif loss_name in 'density':
            if batch['points'].sum() == 0:
                density = 0
            else:
                import kornia
                sigma=1
                kernel_size = (3, 3)
                sigma_list = (sigma, sigma)
                gfilter = kornia.filters.get_gaussian_kernel2d(kernel_size, sigma_list).cuda()
                density = kornia.filters.filter2D(batch['points'][None].float().cuda(), kernel=gfilter[None], border_type='reflect')

            logits = self.model_base(images)
            diff = (logits - density)**2
            loss = torch.sqrt(diff.mean())

        elif loss_name == 'lcfcn_consistency':
            # implementation needed
            logits = self.model_base(images)
            loss = lcfcn_loss.compute_loss(points=batch['points'], probs=logits.sigmoid())
            
            logits_flip = self.model_base(flips.Hflip()(images))
            loss_const = torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))
          
            loss += loss_const

        elif loss_name in 'lcfcn_rot_loss':
            # implementation needed
            logits = self.model_base(images)
            loss = lcfcn_loss.compute_loss(points=batch['points'], probs=logits.sigmoid())
            
            rotations = np.random.choice([0, 90, 180, 270], images.shape[0], replace=True)
            images = flips.Hflip()(images)
            images_rotated = sst.batch_rotation(images, rotations)
            logits_rotated = self.model_base(images_rotated)
            logits_recovered = sst.batch_rotation(logits_rotated, 360 - rotations)
            logits_recovered = flips.Hflip()(logits_recovered)
            
            loss += torch.mean(torch.abs(logits_recovered-logits))
            
        if loss != 0:
            loss.backward()
            if self.exp_dict['model'].get('clip_grad'):
                ut.clip_gradient(self.opt, 0.5)
            try:
                self.opt.step()
            except:
                self.opt.step(loss=loss)

        return {'train_loss': float(loss)}

    @torch.no_grad()
    def predict_on_batch(self, batch, method='semseg'):
        self.eval()
        image = batch['images'].cuda()
        if self.exp_dict['model'].get('4D_input'):
            image = torch.cat([image, batch["ndvi"][0].cuda()[None,None]], dim=1)

        logits = self.model_base.forward(image)
        if method == 'semseg':
            if self.n_classes == 1:
                
                if 'shape' in batch['meta'][0]:
                    logits = F.upsample(logits, size=batch['meta'][0]['shape'],              
                                mode='bilinear', align_corners=False)
                res = (logits.sigmoid().data.cpu().numpy() > 0.5).astype('float')
            else:
                self.eval()
                res = logits.argmax(dim=1).data.cpu().numpy()
        
        if method == 'counts':
            if self.exp_dict['model']['loss'] == 'lcfcn':
                blobs = lcfcn_loss.get_blobs(np.array(logits.sigmoid().squeeze().cpu()))
                points = lcfcn_loss.blobs2points(blobs)
                res = float(points.sum())
            else:
                res = float(logits.sum())

        if method == 'points':
            if self.exp_dict['model']['loss'] in ['lcfcn','lcfcn_consistency', 'lcfcn_nopretrain']:
                blobs = lcfcn_loss.get_blobs(np.array(logits.sigmoid().squeeze().cpu()))
                points = lcfcn_loss.blobs2points(blobs)
                res = points[None]

            elif self.exp_dict['model']['loss'] == 'density':
                res = logits.cpu().numpy()
                
            elif self.exp_dict['model']['loss'] in ['prm']:
                res = get_peak_map(logits,
                                    win_size=3,
                                    counts=None).cpu().numpy()
                print('res', res.sum())
            elif self.exp_dict['model']['loss'] in ['prm_points']:
                res = get_peak_map(logits,
                                    win_size=3,
                                    counts=None).cpu().numpy()
                print('res', res.sum())
            elif self.exp_dict['model']['loss'] in ['cam']:
                res = np.zeros(logits.shape)
                val = float(logits.mean().sigmoid() > 0.5)
                res[0,0,0,0] = val
                assert(res.sum()<= 1)
                # print('res', res.sum())
        return res 

    def vis_on_batch(self, batch, savedir_image):
        image = batch['images']
        original = hu.denormalize(image, mode='rgb')[0]
        img_pred = hu.save_image(savedir_image,
                    original,
                      mask=self.predict_on_batch(batch, method='semseg'), return_image=True)

        img_gt = hu.save_image(savedir_image,
                     original,
                     return_image=True)
                     
        gt_counts = float(batch['points'].sum())
        pred_counts = self.predict_on_batch(batch, method='counts')
        img_gt = models.text_on_image( 'Groundtruth: %d' % gt_counts, np.array(img_gt), color=(0,0,0))
        img_pred = models.text_on_image( 'Prediction: %d' % pred_counts, np.array(img_pred), color=(0,0,0))
        
        if 'points' in batch:
            pts = (batch['points'][0].numpy().copy() != 0).astype('uint8')

            img_gt = np.array(hu.save_image(savedir_image, img_gt/255.,
                                points=pts.squeeze(), radius=2, return_image=True))

        img_list = [np.array(img_gt), np.array(img_pred)]
        hu.save_image("%s" %(savedir_image), np.hstack(img_list))

    @torch.no_grad()
    def vis_on_loader(self, loader, savedir_images):
        self.eval()
        for i, batch in enumerate(tqdm.tqdm(loader)):
            self.vis_on_batch(batch, "%s_%04d.png" %(savedir_images, i))


    def val_on_loader(self, loader, savedir_images=None, n_images=0, keep_counts=False):
        self.eval()
        val_meter = metrics.CountMeter(split=loader.dataset.split, keep_counts=keep_counts)
        i_count = 0
        
        for i, batch in enumerate(tqdm.tqdm(loader)):
            gt_points = batch["points"]
            pred_points = self.predict_on_batch(batch, method='points')
            
            val_meter.update(pred_points=pred_points, gt_points=gt_points)
            if i_count < n_images and len(batch['point_list'][0]) > 0:
                self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images, 
                    '%d.png' % batch['meta'][0]['index']))
                i_count += 1
        
        return val_meter.get_avg_score()
        
    @torch.no_grad()
    def compute_uncertainty(self, images, replicate=False, scale_factor=None, n_mcmc=20, method='entropy'):
        self.eval()
        set_dropout_train(self)

        # put images to cuda
        images = images.cuda()
        _, _, H, W= images.shape

        if scale_factor is not None:
            images = F.interpolate(images, scale_factor=scale_factor)
        # variables
        input_shape = images.size()
        batch_size = input_shape[0]

        if replicate and False:
            # forward on n_mcmc batch      
            images_stacked = torch.stack([images] * n_mcmc)
            images_stacked = images_stacked.view(batch_size * n_mcmc, *input_shape[1:])
            logits = self.model_base(images_stacked)
            

        else:
            # for loop over n_mcmc
            logits = torch.stack([self.model_base(images) for _ in range(n_mcmc)])
            logits = logits.view(batch_size * n_mcmc, *logits.size()[2:])

        logits = logits.view([n_mcmc, batch_size, *logits.size()[1:]])
        _, _, n_classes, _, _ = logits.shape
        # binary do sigmoid 
        if n_classes == 1:
            probs = logits.sigmoid()
        else:
            probs = F.softmax(logits, dim=2)

        if scale_factor is not None:
            probs = F.interpolate(probs, size=(probs.shape[2], H, W))

        self.eval()

        if method == 'entropy':
            score_map = - xlogy(probs).mean(dim=0).sum(dim=1)

        if method == 'bald':
            left = - xlogy(probs.mean(dim=0)).sum(dim=1)
            right = - xlogy(probs).sum(dim=2).mean(0)
            bald = left - right
            score_map = bald


        return score_map 



class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}

def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))

@torch.no_grad()
def get_peak_map(ram, win_size=3, counts=None):
    offset = (win_size - 1) // 2
    padding = torch.nn.ConstantPad2d(offset, float('-inf'))
    padded_maps = padding(ram)

    n, c, h, w = padded_maps.size()
    assert(n==1)
    if offset == 0:
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)
    else:
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
    element_map = element_map.to(ram.device)
    
    _, indices = F.max_pool2d(
        padded_maps,
        kernel_size=win_size,
        stride=1,
        return_indices=True)
    
    peak_map = (indices == element_map).float()
    act_peak_map = (ram * peak_map)
    
    n, c, h, w = peak_map.size()
    # peak filtering
    if counts:
        # pick top k
        act_list, ind_list = act_peak_map.view(n, c, -1).topk(k=int(counts.item()), dim=2)
        peak_map = peak_map.view(n, c, -1) * 0
        peak_map[:, :, ind_list] = 1
        peak_map = peak_map.view(n, c, h, w)
    else:
        # apply median
        mask = ram >= apply_median_filter(ram)
        peak_map = (peak_map * mask.float())
    
    return peak_map.float().detach()

def apply_median_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)