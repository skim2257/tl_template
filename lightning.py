import os
import psutil
import nrrd
import glob
import math
import time
from copy import deepcopy

from typing import Callable, Optional, Tuple
from argparse import ArgumentParser, Namespace
from warnings import warn
from copy import copy
from itertools import islice

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
from torch import nn
from torch.nn import functional as F
# from torch.nn.modules.pooling import AdaptiveAvgPool3d
from torch.utils.data import DataLoader#, random_split, Subset, Dataset
# from torchvision.datasets import MNIST
# from torch.autograd import Variable
from adabound import AdaBound
# from math import floor, pi

# from sklearn.model_selection import train_test_split
from sejin.utils import metrics

import pytorch_lightning as pl

from dataset import StoicDataset
from models.acsconv import ACSModel
from models.resnet import resnet10, resnet18, resnet34, resnet50
from models.med3d import generate_model
from transforms import *

class StoicModel(pl.LightningModule):
    """
    pENE LightningModule class with relevant train/val/test steps and dataloaders.
    """

    def __init__(self, params: Namespace = None):
        """
        Parameters
        ----------
        params
            `Namespace` object containing the model hyperparameters.
            Should usually be generated automatically by `argparse`.
        """
        super(StoicModel, self).__init__()
        
        self.params = params
        self.last = None
        self.acsconv = False
        try:
            self.v = params.verbose
        except:
            self.v = False

        d, h, w = params.input_size
        if params.arch == 'scratch': #end-to-end
            if params.resnet == 'resnet10':
                self.model = resnet10(sample_input_D=d, sample_input_H=h, sample_input_W=w)
            elif params.resnet == 'resnet18':
                self.model = resnet18(sample_input_D=d, sample_input_H=h, sample_input_W=w)
            elif params.resnet == 'resnet34':
                self.model = resnet34(sample_input_D=d, sample_input_H=h, sample_input_W=w)
            elif params.resnet == 'resnet50':
                self.model = resnet50(sample_input_D=d, sample_input_H=h, sample_input_W=w)
        elif params.arch == 'acsconv': #end-to-end
            self.acsconv = True
            self.model = ACSModel(pretrained=params.pretrained, act=params.activation, dropout=params.dropout, resnet=params.resnet)
        elif params.arch == 'med3d': #end-to-end
            pretrain_path = os.path.join(params.med3d_path, f"{params.resnet}.pth")
            self.model = generate_model(input_D=d, input_H=h, input_W=w, resnet=params.resnet, pretrained=params.pretrained, pretrain_path=pretrain_path)
            # self.model = DDDCustom(start_pow=params.start_pow, dropout=params.dropout, n_blocks=params.n_blocks, res=params.res, input_size=params.input_size, resnet=params.resnet)
        else:
            pass
            # self.model = DDDSimple(start_pow=params.start_pow, dropout=params.dropout)

        # LOSS function
        if self.params.loss_fn.lower() == 'bce': 
            self.loss = F.binary_cross_entropy
        elif self.params.loss_fn.lower() == 'l1':
            self.loss = F.l1_loss
        elif self.params.loss_fn.lower() == 'mse':
            self.loss = F.mse_loss
        elif self.params.loss_fn.lower() == 'softmargin':
            self.loss = F.soft_margin_loss
        elif self.params.loss_fn.lower() == 'nll':
            self.loss = F.nll_loss  # RuntimeError: Expected object of scalar type Long but got scalar type\
                                    # Float for argument #2 'target' in call to _thnn_nll_loss_forward
        
        print(self.model)
        if not self.params.pretrained and not self.params.freeze:
            self.init_params(self.model)

    def forward(self, x):
        """
        Forward prop on samples
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss, 'y_hat': y_hat.detach(), 'y': y.detach()}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'y_hat': y_hat}
    
    def training_epoch_end(self, outputs):
        # current_params = torch.clone(next(islice(self.parameters(), 0, None))).to('cpu')
        # print(torch.all(self.first_params == current_params))
        
        
        loss    = torch.stack([j['loss'] for j in outputs]).mean()
        y       = torch.cat([j['y'] for j in outputs]).cpu()
        y_hat   = torch.cat([j['y_hat'] for j in outputs]).cpu()

        y_covid, y_severe         = y[:,0], y[:,1]
        y_hat_covid, y_hat_severe = y_hat[:,0], y_hat[:,1]

        acc, auroc, auprc    = metrics(y_severe, y_hat_severe)
        acc1, auroc1, auprc1 = metrics(y_covid, y_hat_covid)
            
        self.log_dict({'train_loss': loss,
                       'train_auroc_severe': auroc,
                       'train_auprc_severe': auprc,
                       'train_acc_severe': acc,
                       'train_auroc_covid': auroc1,
                       'train_auprc_covid': auprc1,
                       'train_acc_covid': acc1}, prog_bar=True) 
    
    def validation_epoch_end(self, outputs):
        loss     = torch.stack([j['loss'] for j in outputs]).mean()
        y        = torch.cat([j['y'] for j in outputs]).cpu()
        y_hat    = torch.cat([j['y_hat'] for j in outputs]).cpu()
        
        y_covid, y_severe         = y[:,0], y[:,1]
        y_hat_covid, y_hat_severe = y_hat[:,0], y_hat[:,1]

        acc, auroc, auprc    = metrics(y_severe, y_hat_severe)
        acc1, auroc1, auprc1 = metrics(y_covid, y_hat_covid)
        
        if self.v:
            print("\n", y_hat)
            print("\n", y)
            
        self.log_dict({'val_loss': loss,
                       'val_auroc_severe': auroc,
                       'val_auprc_severe': auprc,
                       'val_acc_severe': acc,
                       'val_auroc_covid': auroc1,
                       'val_auprc_covid': auprc1,
                       'val_acc_covid': acc1}, prog_bar=True) 
    
    def test_epoch_end(self, outputs):
        y_hat = torch.cat([j['y_hat'] for j in outputs]).cpu()
        mrns = self.test_dataset.clinical_data.PatientID
        if self.v: print(y_hat.shape, len(mrns))
        pd.DataFrame.from_dict({'MRN': mrns, 'covid': y_hat[:,0], 'severe': y_hat[:,1]}).to_csv(self.params.pred_save_path, index=False)
    
    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.

        Parameters
        ----------
        m
            The module to initialize.

        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.

        References
        ----------
        .. [1] K. He et al. `Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification`,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

        self.first_params = torch.clone(next(islice(self.parameters(), 0, None))).to('cpu')

    def prepare_data(self):
        """Preprocess the data and create training, validation and test
        datasets.

        This method is called automatically by pytorch-lightning.
        """
        
        train_dataset = StoicDataset(self.params.root_directory, 
                                     self.params.clinical_data_path,
                                     input_size=self.params.input_size,
                                     dataaug=self.params.dataaug,
                                     num_workers=self.params.num_workers,
                                     acsconv=self.acsconv,
                                     mode=self.params.input_mode,
                                     organ=self.params.organ,
                                     norm=self.params.norm,
                                     split='train',
                                     train_fold=self.params.train_fold)
        val_dataset   = StoicDataset(self.params.root_directory, 
                                     self.params.clinical_data_path, 
                                     input_size=self.params.input_size,
                                     num_workers=self.params.num_workers,
                                     acsconv=self.acsconv,
                                     mode=self.params.input_mode,
                                     organ=self.params.organ,
                                     norm=self.params.norm,
                                     split='val',
                                     train_fold=self.params.train_fold)
        test_dataset   = StoicDataset(self.params.root_directory, 
                                self.params.clinical_data_path, 
                                input_size=self.params.input_size,
                                num_workers=self.params.num_workers,
                                acsconv=self.acsconv,
                                mode=self.params.input_mode,
                                organ=self.params.organ,
                                norm=self.params.norm,
                                split='test',
                                train_fold=self.params.train_fold)

        # ensuring balanced train/val split
        full_targets = train_dataset.clinical_data["probSevere"]
        
        self.pos_weight = torch.tensor(len(full_targets) / sum(full_targets))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        print('Train/Val/Test:', len(train_dataset), len(val_dataset))#, len(test_dataset))
        
    def train_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.train_dataset,
                          batch_size=self.params.batch_size,
                          num_workers=self.params.num_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.val_dataset,
                          batch_size=self.params.batch_size,
                          num_workers=self.params.num_workers,
                          shuffle=False)
    
    def test_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.test_dataset,
                          batch_size=self.params.batch_size,
                          num_workers=self.params.num_workers,
                          shuffle=False)
    
    def configure_optimizers(self):
        """This method is called automatically by pytorch-lightning."""
        if self.params.opt.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.params.lr,
                                         weight_decay=self.params.weight_decay)
        elif self.params.opt.lower() == "sgd": 
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.params.lr,
                                        weight_decay=self.params.weight_decay)
        elif self.params.opt.lower() == "adabound":
            optimizer = AdaBound(self.parameters(),
                                 lr=self.params.lr,
                                 )
        
        if self.params.scheduler.lower() == 'plateau':
            scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.05, verbose=self.v),
                         "monitor": "val_loss"}
        elif self.params.scheduler.lower() == 'multistep':
            scheduler = {"scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 300], gamma=0.1, verbose=self.v)}
        elif self.params.scheduler.lower() == 'cosine':
            scheduler = {"scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, verbose=self.v)}
        elif self.params.scheduler.lower() == 'poly':
            scheduler = {"scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.975, verbose=self.v)}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}