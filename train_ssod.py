import sys
sys.path.append('/data/sonnh8/ObjectDetectionMethods/Centernet_pytorch')

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
import os
from argparse import ArgumentParser
import yaml
from PIL import Image
from SSOD.dataset import Generator
from SSOD.ema import ModelEMA

from model import Model
from loss import Loss, UnsupLoss
from evaluate import generate_coco_format_predict, generate_coco_format, evaluate
from utils import save_batch, save_batch_pred
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

class LightningModel(pl.LightningModule):
    def __init__(self, config, resizer=None, assigner=None):
        super().__init__()
        self.example_input_array = torch.Tensor(2, 3, 640, 640)
        self.save_hyperparameters()
        with torch.no_grad(): 
            self.model_teacher = ModelEMA(Model(version=config['version'], nc=config['nc'],
                                                max_boxes=10,
                                                is_training=True, load_pretrained=False), decay=0.9996)
            
        self.model = Model(version=config['version'], nc=config['nc'],
                                    max_boxes=config['max_boxes'], is_training=True,
                                    load_pretrained=True)
        self.config = config
        self.suploss_fn = Loss()
        self.unsuploss_fn = UnsupLoss()
        self.resizer = resizer
        self.assigner = assigner
        self.validation_step_outputs = []
        
        self.start_semi_epoch = 20
                
    def forward(self, x):
        out = self.model(x)
        detections = self.model.decoder(out)
        return detections
    
    def configure_optimizers(self):
        opt_config = self.config['optimizer']
        if opt_config['type'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_config['base_lr'])
        elif opt_config['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=opt_config['base_lr'],
                                         foreach=True)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_config['base_lr'])
        
        scheduler = CosineAnnealingLR(optimizer,
                T_max=self.config['epochs'], eta_min=opt_config['end_lr'])
        
        if opt_config['warmup_epochs'] > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=opt_config['warmup_epochs'])
            scheduler = SequentialLR(optimizer,
                                     schedulers=[warmup_scheduler, scheduler],
                                     milestones=[opt_config['warmup_epochs']])
            
            if not hasattr(scheduler, "optimizer"):      # https://github.com/pytorch/pytorch/issues/67318
                setattr(scheduler, "optimizer", optimizer)
        
        return {'optimizer':optimizer, 'lr_scheduler':scheduler}
    
    def on_train_epoch_start(self):
        self.model_teacher.model.eval()
        self.model_teacher.model.to(self.device)
        self.model_teacher.model.half()
    
    def training_step(self, batch, batch_idx):
        unsup_weak_images, unsup_strong_images, images, targets = batch
        unsup_weak_images = torch.cat(unsup_weak_images, 0)
        unsup_strong_images = torch.cat(unsup_strong_images, 0)

        # Supervised training 
        images = images.permute(0, 3, 1, 2).contiguous() #transpose image to 3xHxC
        output = self.model(images)
        
        sup_loss, loss_dict = self.suploss_fn(output, targets)
       
        # Start unsupervised training from epoch 20
        if self.current_epoch > self.start_semi_epoch:
            unsup_weak_images = unsup_weak_images.permute(0, 3, 1, 2).contiguous()
            unsup_strong_images = unsup_strong_images.permute(0, 3, 1, 2).contiguous()
            student_output = self.model(unsup_strong_images)
            with torch.no_grad():
                teacher_output = self.model_teacher.model(unsup_weak_images)
#                 ## Thử theo hướng pseudo-labels trước xem kết quả thế nào
#                 hm_targets, wh_targets, reg_targets, indices_targets  = [], [], [], []
#                 for pred in preds:
#                     threshold =  min(pred[0][4], max(0.05 + (0.2-0.05)*self.current_epoch/100, float(pred[-1][4])))
#                     pred = np.array([p for p in pred if p[4] >= threshold])
#                     hm, wh, reg, indices = self.assigner(pred[:, :4], pred[:, 5].astype('int32'))
#                     # print(sum)
#                     hm_targets.append(hm)
#                     wh_targets.append(wh)
#                     reg_targets.append(reg)
#                     indices_targets.append(indices)
                
#                 unsup_targets = [torch.from_numpy(np.stack(hm_targets, 0)).to(self.device).half(),
#                                  torch.from_numpy(np.stack(wh_targets, 0)).to(self.device).half(),
#                                  torch.from_numpy(np.stack(reg_targets, 0)).to(self.device).half(),
#                                  torch.from_numpy(np.stack(indices_targets, 0)).to(self.device).half()]
                    
            unsup_loss, unsup_loss_dict = self.unsuploss_fn(student_output, teacher_output, self.current_epoch)
            
            if batch_idx < 5:
                preds = self.model_teacher.model.decoder(teacher_output).cpu().numpy() #Nx10x6
                # print(teacher_output[1][0][:, :10, :10])
                # assert False
                save_batch_pred(unsup_weak_images.permute(0, 2, 3, 1).contiguous().cpu().numpy(), 
                                preds, size=640, save_dir=self.config['save_dir'],
                                name=str(batch_idx)+'_%d.jpg'%self.current_epoch)
                
            loss_dict.update(unsup_loss_dict)
            
        # EMA update for teacher
        if self.current_epoch == self.start_semi_epoch-5:
            self.model_teacher.update(self.model, decay=0.0)
            
        elif self.current_epoch > self.start_semi_epoch-5:
            self.model_teacher.update(self.model)
                    
        for key in loss_dict:
            self.log(key, loss_dict[key].item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist =True)
        
        if self.current_epoch > self.start_semi_epoch:
            return sup_loss + 0.1*unsup_loss 
        
        return sup_loss
    
    def validation_step(self, val_batch, batch_idx):
        # pass
        images, impaths = val_batch
        images = images.permute(0, 3, 1, 2).contiguous() #transpose image to 3xHxC
        
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
        if self.global_rank == 0:
            total_output = np.stack(self.validation_step_outputs)
            pred_path = os.path.join(self.config['save_dir'], 'val_predictions.json')
            label_path = os.path.join(config['save_dir'], 'val_labels.json')
            generate_coco_format_predict(total_output, pred_path)
            stats = evaluate(label_path, pred_path)
            val_metrics = {'mAP50':stats[1], 'mAP50-95':stats[0]}
            print('mAP@50    : %8.3f'%stats[1])
            print('mAP@50:95 : %8.3f'%stats[0])
            # val_metrics = {'mAP50':0, 'mAP50-95':0}
            self.log_dict(val_metrics, prog_bar=False, on_step=False, on_epoch=True, sync_dist =True)
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
                           num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset, shuffle=False, batch_size=config['batch_size'], num_workers=4)

generate_coco_format(val_dataset, os.path.join(config['save_dir'], 'val_labels.json'))

if __name__ == '__main__':
    ## Load model
    model = LightningModel(config, resizer=val_dataset.resizer, assigner=train_dataset.assigner)

    pbar = RichProgressBar(refresh_rate=1, leave=True)
    ckpt = ModelCheckpoint(dirpath=config['save_dir'], filename='{epoch}_{mAP50:.3f}',
                           monitor='mAP50', mode='max', save_last=True, save_top_k=3,
                           every_n_epochs=1)

    trainer = pl.Trainer(max_epochs=config['epochs'], default_root_dir=config['save_dir'], #profiler='simple',
                            accelerator="gpu", devices=config['gpu'], precision='16-mixed',
                            callbacks=[pbar, ckpt], limit_train_batches=1000)
    #Resume if needed
    if config['resume'] and config['checkpoint']:
        if os.path.exists(config['checkpoint']):
            ckpt_path = config['checkpoint']
            print('Resume from checkpoint: ', config['checkpoint'])
        else:
            ckpt_path = None
            print('Checkpoint %s not found !'%config['checkpoint'])
    else: 
        ckpt_path = None
            
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
