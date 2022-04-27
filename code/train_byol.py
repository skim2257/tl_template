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

from args_byol import parser

from byol.nlstmodule import NLSTDataModule
from byol.ssleval import SSLOnlineEvaluator
from byol.transforms import NLSTTrainDataTransform, NLSTEvalDataTransform, NLSTFinetuneTransform
from byol_lightning import BYOL


print("Starting...")
# Cleaning up some hparams
hparams = parser()

if hparams.seed is None:
    hparams.seed = np.random.randint(1, 99999)

np.seterr(divide='ignore', invalid='ignore')
print(os.environ)

# get slurm version
slurm_id = os.environ.get("SLURM_JOB_ID")
if slurm_id is None:
    version = None
else:
    version = str(slurm_id)
print(version)

# configure callbacks
checkpoint = ModelCheckpoint(filename="pENE_{epoch:03d}_{val_loss:.3f}_{train_loss:.3f}",
                            save_top_k=5,
                            monitor="val_loss",
                            mode="min")
lr_monitor = LearningRateMonitor(logging_interval="step")
all_callbacks = [checkpoint, lr_monitor]

logger = CometLogger(project_name="BYOL",
                     workspace='skim2257',
                     experiment_name=version,
                     save_dir=hparams.cometdir,
                     offline=True)

ckptpath = os.path.join(hparams.logdir, hparams.exp_name, version)
if not os.path.exists(ckptpath):
    os.makedirs(ckptpath)
print("checkpoint path:", ckptpath)


# init default datamodule
dm = NLSTDataModule.from_argparse_args(hparams)
dm.train_transforms = NLSTTrainDataTransform()
dm.val_transforms = NLSTEvalDataTransform()
model = BYOL(**hparams.__dict__)

# finetune in real-time
online_eval = SSLOnlineEvaluator(z_dim=2048, num_classes=dm.num_classes)

trainer = Trainer.from_argparse_args(hparams, 
                                     logger=[logger], 
                                     callbacks=[online_eval, checkpoint, lr_monitor],
                                     weights_save_path=ckptpath)

trainer.fit(model, datamodule=dm)
print(checkpoint.best_model_path)

print("We've reached the end...")

