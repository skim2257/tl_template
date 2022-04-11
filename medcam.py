from medcam import medcam
# from pytorch_grad_cam.utils.image import show_cam_on_image

import os
from argparse import ArgumentParser
import errno
import numpy as np
import SimpleITK as sitk

from comet_ml import OfflineExperiment

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from lightning import StoicModel
from dataset import StoicDataset
from args import parser

print("Starting...")

# Cleaning up some hparams
hparams = parser()

# Set seed
if hparams.seed is None:
    hparams.seed = np.random.randint(1, 99999)

# Freeze
hparams.freeze = True

# Input size to tuple
hparams.input_size = tuple(hparams.input_size)
print(hparams)

seed_everything(hparams.seed)
np.seterr(divide='ignore', invalid='ignore')

# get slurm version
slurm_id = os.environ.get("SLURM_JOBID")
if slurm_id is None:
    version = None
else:
    version = str(slurm_id)

# init model
print(hparams.ckpt_path)
model = StoicModel.load_from_checkpoint(hparams.ckpt_path, params=hparams).cuda()
model.prepare_data()
print(model.hparams)
print("\n\n\n")
print(model)
# if hparams.arch == 'acsconv':
#     target_layer = model.model.model_3d.layer3[0].conv2
#     cam = GradCAM(model=model.model.model_3d, target_layer=target_layer, use_cuda=True)
# elif hparams.arch == 'med3d':
#     target_layer = model.model.layer4[-1]
#     cam = GradCAM(model=model.model, target_layer=target_layer, use_cuda=True)
# input_tensor = # Create an input tensor image for your model..

# # Construct the CAM object once, and then re-use it on many images:

# train_image = torch.zeros(3, 128, 128, 128)
train_image, _ = model.train_dataset[25]
test_image, _ = model.test_dataset[35]
print(test_image.shape, test_image.max(), test_image.min())

train_image = torch.unsqueeze(train_image, axis=0).cuda()
test_image = torch.unsqueeze(test_image, axis=0).cuda()

print(train_image.shape, test_image.shape)
try:
    os.makedirs(hparams.pred_save_path)
except:
    pass

if hparams.arch == 'acsconv':
    layers = ['model_3d.layer4', \
            'model_3d.layer4.1.conv2', \
            'model_3d.layer4.1.conv1', \
            'model_3d.layer4.0.conv2', \
            'model_3d.layer4.0.conv1', \
            'model_3d.layer3',\
            'model_3d.layer3.1.conv2', \
            'model_3d.layer3.1.conv1', \
            'model_3d.layer3.0.conv2', \
            'model_3d.layer3.0.conv1', \
            'model_3d.layer2', \
            'model_3d.layer1']
elif hparams.arch == 'med3d':
    layers = ['layer4', \
            'layer4.1.conv2',\
            'layer4.1.conv1',\
            'layer4.0.conv2',\
            'layer4.0.conv1',\
            'layer3',\
            'layer3.1.conv2',\
            'layer3.1.conv1',\
            'layer3.0.conv2',\
            'layer3.0.conv1',\
            'layer2', 
            'layer1']

train_path = os.path.join(hparams.pred_save_path, 'train_img')
try:
    os.makedirs(train_path)
except:
    pass
model.model = medcam.inject(model.model, 
                            backend='gcam',
                            layer=layers, 
                            output_dir=train_path, 
                            save_maps=True)
output = model(train_image)


test_path = os.path.join(hparams.pred_save_path, 'train_img')
try:
    os.makedirs(test_path)
except:
    pass
model.model = medcam.inject(model.model, 
                            backend='gcam',
                            layer=layers, 
                            output_dir=test_path, 
                            save_maps=True)
output = model(test_image)

# # Initialize a trainer
# trainer = Trainer.from_argparse_args(hparams, 
#                                      progress_bar_refresh_rate=2,
#                                      checkpoint_callback=None,
#                                      logger=None,
#                                      deterministic=True)                                     


print("We've reached the end...")

  