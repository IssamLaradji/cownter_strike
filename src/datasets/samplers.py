import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from torch.utils.data import sampler

class BalancedSampler(sampler.Sampler):
    def __init__(self, data_source, n_samples):
        self.data_source = data_source
        self.n_samples = n_samples
        self.n = len(self.data_source)
        self.nf = (self.data_source.labels!=0).sum()
        self.nb = (self.data_source.labels==0).sum()

        self.nb_ind = (self.data_source.labels==0)
        self.nf_ind = (self.data_source.labels!=0)
        
    def __iter__(self):
        p = np.ones(len(self.data_source))
        p[self.nf_ind] =  1./ self.nf 
        p[self.nb_ind] =  1./self.nb
        p = p / p.sum()

        indices = np.random.choice(np.arange(self.n), 
                                   self.n_samples, 
                                   replace=True, 
                                   p=p)

        return iter(indices)

    def __len__(self):
        return self.n_samples