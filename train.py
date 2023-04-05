import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
from dataset.generator import Generator
from model import Model
from torchinfo import summary

parser = ArgumentParser()
parser.add_argument('--config', default='config/default.yaml')
parser.add_argument('--gpu', default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

## Load config
with open(args.config) as f:
    cfg = yaml.safe_load(f)

## Load data [Done]
# train_dataset = Generator(cfg, mode='train')
# val_dataset   = Generator(cfg, mode='val')
# train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=cfg['batch_size'], num_workers=4)
# val_loader    = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=4)

## Load model
model = Model(version=cfg['version'], nc=cfg['nc'], is_training=False)
summary(model, input_size=(1, 3, cfg['input_size'], cfg['input_size']))


