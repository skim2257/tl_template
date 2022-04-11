import os
import argparse
from dotenv import load_dotenv
from pytorch_lightning import Trainer

def parser():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    load_dotenv(".env")
    
    # Model hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="The initial learning rate.")
    parser.add_argument("--input_size", type=int, nargs='+', default=[64, 64, 64], help="Size of cropped volume.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay of lr.")
    parser.add_argument("--activation", type=str, default='relu', help="Type of activation function.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout in fc layers.")
    parser.add_argument("--arch", type=str, default='dddcustom', help="Model architecture of choice.")
    parser.add_argument("--resnet", type=str, default='18', help="ResNet architecture of choice.")
    
    parser.add_argument("--loss_fn", type=str, default='bce', help="Specify loss function.")
    parser.add_argument("--opt", type=str, default='adam', help="Type of optimizer.")
    parser.add_argument("--scheduler", type=str, default='plateau', help="Type of LR Scheduler.")

    parser.add_argument("--start_pow", type=int, default=5, help="Power of 2 for starting number of channels.")
    parser.add_argument("--n_blocks", type=int, default=3, help="Number of blocks of Conv3d layers.")
    parser.add_argument("--res", type=bool, default=False, help="Whether or not to make DDDCustom blocks residual")
    parser.add_argument("--input_mode", type=str, default='box', help="Mode of input: ['box', 'small', 'raw_img', 'box_nomask', 'small_nomask']")
    parser.add_argument("--organ", type=str, default='lung', help="Which organ to optimize HU range for?")
    parser.add_argument("--norm", type=str, default='minmax', help="Which normalization option?")

    # janky stuff
    parser.add_argument("--forward_size", type=int, default=1, help="Number of samples per mini batch for Checkpointing. Currently only used for 3D CNNs.")
    parser.add_argument("--train_fold", type=int, default=1, help="Fold # for train/val split.")
    
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument("--freeze", dest="freeze", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--noaug", dest="dataaug", action="store_false")
    parser.add_argument("--softacs", dest="softacs", action="store_true")
    parser.add_argument("--nolog", dest="nolog", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")

    # Training    
    parser.add_argument("--root_directory", type=str, default=os.getenv("ROOT_PATH"), help="Directory of images/masks.")
    parser.add_argument("--clinical_data_path", type=str, default=os.getenv("CLIN_PATH"), help="Path to clinical data file.")
    parser.add_argument("--logdir", type=str, default=os.getenv("LOG_PATH"), help="Directory where training logs/checkpoints will be saved.")
    parser.add_argument("--pred_save_path", type=str, default="./predictions.csv", help="Final prediction data path.")
    parser.add_argument("--sample_idx", type=str, default=0, help="Path of sample to run GradCAM.")
    parser.add_argument("--cometdir", type=str, default=os.getenv('COMET_PATH'), help="Directory where comet logs will be saved")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Model checkpoint file path.")
    parser.add_argument("--med3d_path", type=str, default=os.getenv("MED3D_PATH"), help="Med3d save path.")
    
    parser.add_argument("--num_workers", type=int, default=4, help="# of subprocesses.")
    parser.add_argument("--exp_name", type=str, default="pENE_nodes", help="Experiment name.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for RNG.")
    
    parser.set_defaults(pretrained=False, nolog=False, dataaug=True, freeze=False, softacs=False, verbose=False)
    return parser.parse_known_args()[0]

    
    