import os
import numpy as np

from pytorch_lightning import Trainer, seed_everything

from lightning import StoicModel
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
model = StoicModel.load_from_checkpoint(hparams.ckpt_path, params=hparams).cuda()

# Initialize a trainer
trainer = Trainer.from_argparse_args(hparams, 
                                     progress_bar_refresh_rate=2,
                                     checkpoint_callback=None,
                                     logger=None)

# Train the model ⚡
trainer.test(model)

print("We've reached the end...")

  