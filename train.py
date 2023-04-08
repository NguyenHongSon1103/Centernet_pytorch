import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from argparse import ArgumentParser
import yaml
from dataset.generator import Generator
from model import Model
from torchinfo import summary
from loss import Loss
from trainer import BaseTrainer

parser = ArgumentParser()
parser.add_argument('--config', default='config/default.yaml')
args = parser.parse_args()

## Load config
with open(args.config) as f:
    cfg = yaml.safe_load(f)

cfg['save_dir'] = os.path.abspath(cfg['save_dir'])
os.makedirs(cfg['save_dir'], exist_ok=True)

# os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu'])
## Load data [Done]
train_dataset = Generator(cfg, mode='train')
val_dataset   = Generator(cfg, mode='val')
train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=cfg['batch_size'], num_workers=0)
val_loader    = DataLoader(val_dataset, shuffle=False, batch_size=cfg['batch_size'], num_workers=0)

if not os.path.exists(os.path.join(cfg['save_dir'], 'val_labels.json')):
    val_dataset.generate_coco_format(os.path.join(cfg['save_dir'], 'val_labels.json'))

## Load model
model = Model(version=cfg['version'], nc=cfg['nc'], max_boxes=cfg['max_boxes'], is_training=True)
# device = torch.device('cuda:'+cfg['gpu']) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
model.to(device)

# summary(model, input_size=(1, 3, cfg['input_size'], cfg['input_size']))## Get optimizer and loss
opt_cfg = cfg['optimizer']
optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg['base_lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=cfg['epochs'], eta_min=opt_cfg['end_lr'])

loss_fn = Loss()

trainer = BaseTrainer(model, loss_fn, optimizer, device, train_loader, val_loader, scheduler, cfg['epochs'], cfg)

if __name__ == '__main__':
    trainer.train()
