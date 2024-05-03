import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import dataset
import model
import train
import model_parts as ap
import utils

NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 4

TRAIN_DIR_CONTENT = "data/train_content"
VAL_DIR_CONTENT = "data/val_content"
TRAIN_DIR_STYLE = "data/train_style"
VAL_DIR_STYLE = "data/val_style"

OPTIMIZER = "adam"
LEARNING_RATE = 0.00005
MOMENTUM = 0.6

RESUME_ID = None
CHECKPOINT_FREQ = 1

AMP_TYPE = "bf16"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description='Image colorization using UNet')

parser.add_argument('-e', '--epochs', type=int,
                    default=NUM_EPOCHS,
                    help='Number of training epochs')
parser.add_argument('-bs', '--batch_size', type=int,
                    default=BATCH_SIZE,
                    help='Batch size for training')
parser.add_argument('-nw' ,'--num_workers', type=int,
                    default=NUM_WORKERS,
                    help='Number of workers for data loading')

parser.add_argument('-tdc', '--train_dir_content', type=str,
                    default=TRAIN_DIR_CONTENT,
                    help='Path to the style model train_content image folder')
parser.add_argument('-vdc', '--val_dir_content', type=str,
                    default=VAL_DIR_CONTENT,
                    help='Path to the style model val_content image folder')

parser.add_argument('-tds', '--train_dir_style', type=str,
                    default=TRAIN_DIR_STYLE,
                    help='Path to the style model train_style image folder')
parser.add_argument('-vds', '--val_dir_style', type=str,
                    default=VAL_DIR_STYLE,
                    help='Path to the style model val_style image folder')

parser.add_argument('-op', '--optimizer', type=str,
                    default=OPTIMIZER,
                    help='Optimizer for training',
                    choices=["sgd", "adam"])
parser.add_argument('-lr', '--learning_rate', type=float,
                    default=LEARNING_RATE,
                    help='Learning rate for the optimizer')
parser.add_argument('--momentum', type=float,
                    default=MOMENTUM,
                    help='Momentum for SGD optimizer')

parser.add_argument('-id', '--resume_id', type=str,
                    default=RESUME_ID,
                    help='Wandb run ID to resume training')
parser.add_argument('-cf', '--checkpoint_freq', type=int,
                    default=CHECKPOINT_FREQ,
                    help='Frequency of saving checkpoints during training, -1 for no checkpoints')

parser.add_argument('--enable_dali', action='store_true',
                    help='Enable DALI for faster data loading')
parser.add_argument('--enable_amp', action='store_true',
                    help='Enable Mixed Precision for faster training and lower memory usage')

parser.add_argument('-ampt', '--amp_dtype', type=str,
                    default=AMP_TYPE,
                    help='Set dtype for amp',
                    choices=["bf16", "fp16"])

args = parser.parse_args()

print("Init dataloader")
if args.enable_dali:
    train_loader = DALIGenericIterator(
        [dataset.StyleDataset.dali_pipeline(content_dir=args.train_dir_content,
                                            style_dir=args.train_dir_style,
                                            batch_size=args.batch_size,
                                            num_threads=args.num_workers,
                                            prefetch_queue_depth=4 if args.enable_amp else 2)],
        ['content', 'style'],
        reader_name='Reader'
    )

    val_loader = DALIGenericIterator(
        [dataset.StyleDataset.dali_pipeline(content_dir=args.val_dir_content,
                                            style_dir=args.val_dir_style,
                                            batch_size=args.batch_size,
                                            num_threads=args.num_workers,
                                            prefetch_queue_depth=4 if args.enable_amp else 2)],
        ['content', 'style'],
        reader_name='Reader'
    )
else:
    pass
    
model = model.StyleTransfer()
loss = ap.AdaINLoss()

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

print("Training...")
train.train_model(model, 
                optimizer, 
                loss, 
                train_loader, 
                val_loader, 
                args)
