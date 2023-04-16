'''
Trainer follow by Pytorch_template
'''
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from PIL import Image
from eval import evaluate, evaluate_all
from utils import save_csv, save_batch

class BaseTrainer:
    def __init__(self,  model, loss_fn, optimizer, device,
                data_loader, valid_data_loader=None, test_data_loader=None,
                lr_scheduler=None, config=None):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.data_loader = data_loader
        self.epochs = config['epochs']
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.config = config

        self.save_period = config['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config['save_dir']

        # setup visualization writer instance             
        self.writer = SummaryWriter(os.path.join(self.checkpoint_dir, 'runs'))   
    
        if config['resume']:
            self._resume_checkpoint(config['resume_checkpoint'])
        
        self.loss_keys = ['hm_loss', 'wh_loss', 'reg_loss', 'total_loss']
        self.val_metrics = {'mAP50':[], 'mAP50-95':[]}
        self.resizer = self.data_loader.dataset.resizer
    
    def _write_tensorboard(self, loss_dict, global_step):
        for key in loss_dict:
            self.writer.add_scalar(key, loss_dict[key], global_step=global_step)        

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        
        history_loss = {key:[] for key in self.loss_keys}

        key_nums = len(self.loss_keys) + 2
        print('='*10)
        print('%s    '*key_nums%tuple(['Epoch', 'lr']+self.loss_keys))
        
        pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, (data, target, im_paths) in pbar:
            if epoch == 0:
                #Save first epoch's images:
                save_batch(im_paths, data.numpy(), [tg.numpy() for tg in target],
                 640, self.config['save_dir'], str(batch_idx)+'.jpg')

            data = data.to(self.device).permute(0, 3, 1, 2)
            target = [tg.to(self.device) for tg in target]
            #transpose image to 3xHxC
            self.optimizer.zero_grad()
            output = self.model(data)
            loss, loss_dict = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            #Convert tensor to scalar    
            loss_dict = {key:loss_dict[key].item() for key in loss_dict}

            pbar.set_description('%d    '%epoch + "{:.2e}    ".format(self.optimizer.param_groups[0]['lr']) + \
                                 '%10.4f    '*(key_nums-2)%tuple([loss_dict[key] for key in loss_dict]))

            #Update history loss:
            for key in self.loss_keys:
                history_loss[key].append(loss_dict[key])
            
            #Write tensorboard
            if batch_idx % self.log_step == 0:
                global_step = (epoch - 1) * self.epochs + batch_idx
                self._write_tensorboard(loss_dict, global_step)
            
            


        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        #Get mean history loss for return
        history_loss_mean = {key:np.mean(history_loss[key]) for key in self.loss_keys}
        return history_loss_mean

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        metrics = {'mAP50':0, 'mAP50-95':0}
        all_predictions = []
        with torch.no_grad():
            pbar = tqdm(enumerate(self.valid_data_loader),
                        total=len(self.valid_data_loader),
                        desc='%10s    '*2%('mAP50', 'mAP50-95'))
            
            for batch_idx, (data, target, im_paths) in pbar:
                data = data.to(self.device).permute(0, 3, 1, 2)
                data, target = data.to(self.device), [tg.to(self.device) for tg in target]

                output = self.model(data)
                predictions = self.model.decoder(output).cpu().numpy()
                
                for prediction, im_path in zip(predictions, im_paths):
                    all_predictions.append({'im_path':im_path, 'pred':prediction})
            ## Calculate overall mAP ##
            save_path = os.path.join(self.config['save_dir'], 'val_predictions.json')
            lb_path = os.path.join(self.config['save_dir'], 'val_labels.json')
            self.generate_coco_format_predict(all_predictions, save_path)
            stats = evaluate(lb_path, save_path)
            
            metrics['mAP50'] = stats[1]
            metrics['mAP50-95'] = stats[0]
            ## ----------------------------##

            print('%10.4f    '*2%tuple(metrics[key] for key in metrics))

            #Write tensorboard
            self._write_tensorboard(metrics, epoch)
        return metrics

    def train(self):
        """
        Full training logic
        """
        best_mAP50_95 = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            #Train epoch
            history_loss_mean = self._train_epoch(epoch)
            #Eval epoch
            if self.do_validation:
                self.valid_data_loader.shuffle=False
                metrics = self._valid_epoch(epoch)
                for key in metrics:
                    self.val_metrics[key].append(metrics[key])
            
            #Shuffle data each epoch
            self.data_loader.dataset.on_epoch_end()

            ## save logged informations into log dict and to csv ##
            logs = []
            logs.append({key:round(history_loss_mean[key], 4) for key in history_loss_mean})
            
            for key in self.val_metrics:
                logs[-1][key] = round(self.val_metrics[key][-1], 3)
            logs[-1]['lr'] = float(self.optimizer.param_groups[0]['lr'])
            save_csv(logs[-1], self.config['save_dir'], header=epoch==1)
            ## End save logged ##
            
            ## Save checkpoint ##
            if logs[-1]['mAP50-95'] > best_mAP50_95: 
                best_mAP50_95 = logs[-1]['mAP50-95']
                self._save_checkpoint(epoch, save_best=True, save_last=True)
            else:
                self._save_checkpoint(epoch, save_best=False, save_last=True)
            
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False, save_last=False)
            ## END SAVE CHECKPOINT ##

    def _save_checkpoint(self, epoch, save_best=False, save_last=True):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(state, best_path)
            print("Saving current best: best.pth ...")
        
        if save_last:
            last_path = os.path.join(self.checkpoint_dir, 'last.pt')
            torch.save(state, last_path)
        
        if (not save_best) and (not save_last): #save by period
            filename = os.path.join(self.checkpoint_dir, 'ckpt-epoch{}.pt'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))
        
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            print("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            print("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
    
    def generate_coco_format_predict(self, all_predictions, save_path):
        """
        Use the pycocotools to evaluate a COCO model on a dataset.

        Args
            all_predictions: All predictions on validation set (May be harm if large val set ?)
            save_path: direction to save preditions json file
        """
        # start collecting results
        results = []
        for i, item in enumerate(all_predictions):
            detection = item['pred']
            raw_boxes, scores, class_ids = detection[..., :4], detection[..., 4], detection[..., 5].astype('int32')
            im_w, im_h = Image.open(item['im_path']).size
            raw_boxes = raw_boxes * 4
            boxes = self.resizer.rescale_boxes(raw_boxes, (im_h, im_w))
            name = item['im_path'].split('/')[-1]
            for box, score, class_id in zip(boxes, scores, class_ids):
                xmin, ymin, xmax, ymax = [float(p) for p in box]
                w, h = max(0, xmax - xmin), max(0, ymax - ymin)  

                # append detection for each positively labeled class
                image_result = {
                    'image_id': os.path.splitext(name)[0],
                    'category_id': int(class_id),
                    'score': float(score),
                    'bbox': [xmin, ymin, w, h],
                }
                # append detection to results
                results.append(image_result)

        if not len(results):
            print('No testset found')
            return

        # write output
        with open(save_path, 'w') as f:
            json.dump(results, f)
        # print(f"Prediction to COCO format finished. Resutls saved in {save_path}")
    
    def test(self):
        """
        Validate after training an epoch

        :return: A log that contains information about test
        """
        self.model.eval()
        metrics = {'mAP50':0, 'mAP50-95':0}
        all_predictions = []
        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_data_loader),
                        total=len(self.test_data_loader),
                        desc='%10s    '*2%('mAP50', 'mAP50-95'))

            for batch_idx, (data, _, im_paths) in pbar:
                data = data.to(self.device).permute(0, 3, 1, 2)
                
                predictions = self.model(data)
                # predictions = self.model.decoder(output)[0].cpu().numpy() #Nx100x6
                # boxes, scores, class_ids = predictions[:, :4], predictions[:, 4], predictions[:, 5]

                for prediction, im_path in zip(predictions, im_paths):
                    all_predictions.append({'im_path':im_path, 'pred':prediction})
            ## Calculate overall mAP ##
            save_path = os.path.join(self.config['save_dir'], 'test_predictions.json')
            lb_path = os.path.join(self.config['save_dir'], 'test_labels.json')
            self.generate_coco_format_predict(all_predictions, save_path)
            stats = evaluate_all(lb_path, save_path)

        return stats
