import os
from argparse import ArgumentParser
import errno
import numpy as np

from comet_ml import OfflineExperiment

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from lightning import StoicModel
from args import parser

print("Starting...")
# Cleaning up some hparams
hparams = parser()
if hparams.seed is None:
    hparams.seed = np.random.randint(1, 99999)

hparams.input_size = tuple(hparams.input_size)
print(hparams)

seed_everything(hparams.seed)
np.seterr(divide='ignore', invalid='ignore')

print(os.environ)

# get slurm version
slurm_id = os.environ.get("SLURM_JOB_ID")
if slurm_id is None:
    version = None
else:
    version = str(slurm_id)
print(version)

# configure logger
if hparams.nolog:
    all_loggers, all_callbacks = None, None
else:
    tb_logger = TensorBoardLogger(hparams.logdir, 
                                  name=hparams.exp_name, 
                                  version=version, 
                                  log_graph=True)
    
    logger = CometLogger(project_name="Stoic2021",
                         workspace='skim2257',
                         experiment_name=version,
                         save_dir=hparams.cometdir,
                         offline=True)
    
    logger.log_hyperparams(hparams)
    all_loggers = [logger]

    # configure callbacks
    checkpoint = ModelCheckpoint(filename="pENE_{epoch:03d}_{val_loss:.3f}_{val_auroc_severe:.3f}_{val_auprc_severe:.3f}",
                                save_top_k=5,
                                monitor="val_auroc_severe",
                                mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    all_callbacks = [checkpoint, lr_monitor]

ckptpath = os.path.join(hparams.logdir, hparams.exp_name, version)
if not os.path.exists(ckptpath):
    os.makedirs(ckptpath)
print("checkpoint path:", ckptpath)

# init model
model = StoicModel(hparams)

# Initialize a trainer
trainer = Trainer.from_argparse_args(hparams,
                                     enable_checkpointing=True,
                                     callbacks=all_callbacks,
                                     logger=all_loggers,
                                     weights_save_path=ckptpath,
                                    #  deterministic=True,
                                    #  fast_dev_run=True,  # last two lines are for debugging state only
                                     gpus=1)


# Train the model ⚡
trainer.fit(model)
print(checkpoint.best_model_path)

print("We've reached the end...")

