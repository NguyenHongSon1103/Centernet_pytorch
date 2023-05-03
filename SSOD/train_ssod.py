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

class LightningModel(pl.LightningModule):
    def __init__(self, config, resizer=None):
        super().__init__()
        self.example_input_array = torch.Tensor(2, 3, 640, 640)
        self.save_hyperparameters()
        self.model_teacher = Model(version=config['version'], nc=config['nc'], max_boxes=config['max_boxes'], is_training=True)
        self.model_student = Model(version=config['version'], nc=config['nc'], max_boxes=config['max_boxes'], is_training=True)
        self.config = config
        self.loss_fn = Loss()
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
            optimizer = torch.optim.AdamW(self.parameters(), lr=opt_config['base_lr'])
        else:
            optimizer = Lion(self.parameters(), lr=opt_config['base_lr'])                                                                               
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #         T_max=self.config['epochs'], eta_min=opt_config['end_lr'])
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[50, 80], gamma=0.1)
        
        return {'optimizer':optimizer, 'lr_scheduler':scheduler}
    
    def training_step(self, batch, batch_idx):
        images, targets, impaths = batch
        
        # Save trained images to disk: Default first 100 step
        if batch_idx < 30:
            save_batch(impaths, images.cpu().numpy(), 
                       [tg.cpu().numpy() for tg in targets], blend_heatmap=True,
                       size=self.config['input_size'], save_dir=self.config['save_dir'],
                       name=str(batch_idx)+'_%d.jpg'%self.current_epoch)
        
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
        val_metrics = {'mAP50':stats[1], 'mAP50-95':stats[0]}
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
    ckpt = ModelCheckpoint(dirpath=config['save_dir'], filename='{epoch}_{mAP50:.2f}',
                           monitor='mAP50', mode='max', save_last=True,
                           every_n_epochs=config['save_period'])

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
            
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
