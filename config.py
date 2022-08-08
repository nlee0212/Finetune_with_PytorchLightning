# Standard
from os.path import dirname, abspath, join
import random

# PIP
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Custom


class Config:
    # User Setting
    SEED = 94
    DATASET_DIR = join(dirname(dirname(abspath(__file__))), 'data')  # ~/stock.estimation/data
    TWITTER_DATASET_DIR = join(DATASET_DIR, 'tweet/updown_labelled')
    TWITTER_DATASET_DIR_FOR_FILTERING = join(DATASET_DIR, 'tweet/preprocessed_subsampled_updown_labelled_ours')
    START_TIME = '2014-01-01'  # 시작일 (ex : 1970-01-01)
    END_TIME = '2015-12-31'  # 종료일 (ex : 1970-01-01)

    num_gpus = 1
    max_epochs = 1000
    batch_size = 16
    time_delta = 1 
    criterion = 'RMSE'
    optimizer = 'AdamW'
    learning_rate = 1.6585510780186816e-06
    num_workers = 4
    verbose = 0  # 0: quiet, 1: with log

    def __init__(self, SEED=None):
        if SEED:
            self.SEED = SEED
        self.set_random_seed()

    def set_random_seed(self):
        print(f'=> SEED : {self.SEED}')

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(self.SEED)  # if use multi-GPU

        pl.seed_everything(self.SEED)
