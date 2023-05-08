import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from argparse import ArgumentParser
import yaml
from PIL import Image
from dataset.generator import Generator
# from model_2 import Model
from model import Model
from loss import Loss
from evaluate import generate_coco_format_predict, generate_coco_format, evaluate
from utils import save_batch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

class Lion(torch.optim.Optimizer):
    '''
    Lion optimizer, borrow from: https://github.com/lucidrains/lion-pytorch    
    '''
    def __init__( self, params, lr: float = 1e-4, betas = (0.9, 0.99), weight_decay = 0.0):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict( lr = lr, betas = betas, weight_decay = weight_decay)

        super().__init__(params, defaults)
        
    def update_fn(self, p, grad, exp_avg, lr, wd, beta1, beta2):
        # stepweight decay

        p.data.mul_(1 - lr * wd)

        # weight update

        update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
        p.add_(update, alpha = -lr)

        # decay the momentum running average coefficient

        exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

    @torch.no_grad()
    def step( self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                self.update_fn( p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss

class LightningModel(pl.LightningModule):
    def __init__(self, config, resizer=None):
        super().__init__()
        self.example_input_array = torch.Tensor(2, 3, 640, 640)
        self.save_hyperparameters()
        self.model = Model(version=config['version'], nc=config['nc'], max_boxes=config['max_boxes'], is_training=True)
        self.config = config
        self.loss_fn = Loss(weights=[1.0, 0.2, 1.0]) #increase weights for regression
        self.resizer = resizer
        self.validation_step_outputs = []
        
    def forward(self, x):
        out = self.model(x)
        detections = self.model.decoder(out)
        return detections
    
    def configure_optimizers(self):
        opt_config = self.config['optimizer']
        if opt_config['type'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=opt_config['base_lr'])
        elif opt_config['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=opt_config['base_lr'],
                                         foreach=True)
        else:
            # optimizer = Lion(self.parameters(), lr=opt_config['base_lr'])  
            optimizer = torch.optim.SGD(self.parameters(), lr=opt_config['base_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=self.config['epochs'], eta_min=opt_config['end_lr'])
    
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #             milestones=[50, 80], gamma=0.1)
        
        return {'optimizer':optimizer, 'lr_scheduler':scheduler}
    
    def training_step(self, batch, batch_idx):
        images, targets, impaths = batch
        
        # Save trained images to disk: Default first 100 step
        # if batch_idx < 30:
        #     save_batch(impaths, images.cpu().numpy(), 
        #                [tg.cpu().numpy() for tg in targets], blend_heatmap=True,
        #                size=self.config['input_size'], save_dir=self.config['save_dir'],
        #                name=str(batch_idx)+'_%d.jpg'%self.current_epoch)
        
        images = images.permute(0, 3, 1, 2) #transpose image to 3xHxC
        output = self.model(images)
        loss, loss_dict = self.loss_fn(output, targets)
        for key in loss_dict:
            self.log(key, loss_dict[key].item(), prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        images, targets, impaths = val_batch
        images = images.permute(0, 3, 1, 2) #transpose image to 3xHxC
        # targets = [tg.to(self.device) for tg in targets]
        
        with torch.no_grad():
            out = self.model(images)
            predictions = self.model.decoder(out).cpu().numpy() #Nx100x6
        
        for prediction, im_path in zip(predictions, impaths):
            raw_boxes, scores, class_ids = prediction[..., :4], prediction[..., 4], prediction[..., 5].astype('int32')
            im_w, im_h = Image.open(im_path).size
            raw_boxes = raw_boxes * 4
            boxes = self.resizer.rescale_boxes(raw_boxes, (im_h, im_w))
            self.validation_step_outputs.append({'im_path':im_path, 'boxes':boxes, 'scores':scores, 'class_ids':class_ids})
    
    def on_validation_epoch_end(self):
        total_output = np.stack(self.validation_step_outputs)
        pred_path = os.path.join(self.config['save_dir'], 'val_predictions.json')
        label_path = os.path.join(config['save_dir'], 'val_labels.json')
        generate_coco_format_predict(total_output, pred_path)
        stats = evaluate(label_path, pred_path)
        val_metrics = {'mAP50':stats[1], 'mAP5095':stats[0]}
        print('mAP@50    : %8.3f'%stats[1])
        print('mAP@50:95 : %8.3f'%stats[0])
        self.log_dict(val_metrics, prog_bar=False, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()
        
parser = ArgumentParser()
parser.add_argument('--config', default='config/default.yaml')
args = parser.parse_args()

## Load config
with open(args.config) as f:
    config = yaml.safe_load(f)

## prepair save directory 
config['save_dir'] = os.path.abspath(config['save_dir'])
os.makedirs(config['save_dir'], exist_ok=True)
os.makedirs(os.path.join(config['save_dir'], 'preprocessed'), exist_ok=True)

## Load data
train_dataset = Generator(config, mode='train')
val_dataset   = Generator(config, mode='val')
train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'],
                           num_workers=8, pin_memory=True)
val_loader    = DataLoader(val_dataset, shuffle=False, batch_size=config['batch_size'], num_workers=8)

generate_coco_format(val_dataset, os.path.join(config['save_dir'], 'val_labels.json'))

if __name__ == '__main__':
    ## Load model
    model = LightningModel(config, resizer=val_dataset.resizer)

    pbar = RichProgressBar(refresh_rate=1, leave=True)
    ckpt = ModelCheckpoint(dirpath=config['save_dir'], filename='{epoch}_{mAP50:.3f}',
                           monitor='mAP50', mode='max', save_last=True, save_top_k=3,
                           every_n_epochs=1)

    trainer = pl.Trainer(max_epochs=config['epochs'], default_root_dir=config['save_dir'], profiler='simple',
                            accelerator="gpu", devices=config['gpu'],
                            callbacks=[pbar, ckpt])
    #Resume if needed
    if config['resume'] and config['checkpoint']:
        if os.path.exists(config['checkpoint']):
            ckpt_path = config['checkpoint']
            print('Resume from checkpoint: ', config['checkpoint'])
        else:
            ckpt_path = None
            print('Checkpoint %s not found !'%config['checkpoint'])
            
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
