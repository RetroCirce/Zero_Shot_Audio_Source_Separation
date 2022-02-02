# test model for ddp testing in pytorch lightning
import numpy as np
import librosa
import os
import sys
import math
import pickle
import logging

from utils import get_segment_bgn_end_samples, np_to_pytorch, get_mix_data
from losses import get_loss_func

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.optim as optim
from torch.nn.parameter import Parameter
from torchlibrosa.stft import STFT, ISTFT, magphase
import pytorch_lightning as pl

class TMM(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()
        logging.info("You are using the testing model, this will not train anything.")
        self.fc = nn.Linear(10,10)
        self.dataset = dataset

    def forward(self, x):
        return 1

    def training_step(self, batch, batch_idx):
        # get shape
        logging.info("batch_idx: %d | device: %s | data: %s" %(batch_idx, next(self.parameters()).device, batch))
        return None

    def training_epoch_end(self, outputs):
        self.dataset.get_new_list()
    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(), lr = 1e-4, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
        )