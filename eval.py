from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import xml.etree.ElementTree as ET
import os
from utils import HiddenPrints

def evaluate(anno_json, pred_json):
    # run COCO evaluation
    with HiddenPrints():
        # load results in COCO evaluation tool
        coco_true = COCO(anno_json)
        coco_pred = coco_true.loadRes(pred_json)

        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    #     coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval.stats

def evaluate_all(anno_json, pred_json):
    with HiddenPrints():
        # load results in COCO evaluation tool
        coco_true = COCO(anno_json)
        coco_pred = coco_true.loadRes(pred_json)

        # run COCO evaluation
        stats = []
        for catId in coco_true.getCatIds():
            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.catIds = [catId]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats.append(coco_eval.stats)
    return stats
    
def print_s(stats, num_classes):
    print('-'*5 + ' Evaluate ' + '-'*5)
    print('mAP@50    | ',end='')
    for i in range(num_classes):
        print('Class %d: %8.3f | '%(i, stats[i][1]),end='')
    print('\n')
    print('mAP@50:95 | ',end='')
    for i in range(num_classes):
        print('Class %d: %8.3f | '%(i, stats[i][0]),end='')
    print('\n')
    
def log(stats, num_classes):
    ## Write result to files:
    with open('test_result.txt', 'a+') as f:
        f.write('-'*5 + ' Evaluate ' + '-'*5 + '\n')
        f.write('mAP@50    | ')
        for i in range(num_classes):
            f.write('Class %d: %8.3f | '%(i, stats[i][1]))
        f.write('\n')
        f.write('mAP@50:95 | ')
        for i in range(num_classes):
            f.write('Class %d: %8.3f | '%(i, stats[i][0]))
        f.write('\n')
        
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import numpy as np
    import os
    from argparse import ArgumentParser
    import yaml
    from dataset.generator import Generator
    from model import Model
    from trainer import BaseTrainer

    parser = ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='')
    args = parser.parse_args()

    ## Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg['save_dir'] = os.path.abspath(cfg['save_dir'])
    assert os.path.exists(cfg['save_dir']), 'save folder not found'
    
    if args.device == '-1':
        device = torch.device('cpu')
    else:
        if args.device != '':
            cfg['gpu'] = args.device
        device = torch.device('cuda:'+cfg['gpu']) if torch.cuda.is_available() else torch.device('cpu')

    ## Load data [Done]
    test_dataset = Generator(cfg, mode='test')
    test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=cfg['batch_size'], num_workers=6)

    test_dataset.generate_coco_format(os.path.join(cfg['save_dir'], 'test_labels.json'))

    ## Load model
    model = Model(version=cfg['version'], nc=cfg['nc'], max_boxes=cfg['max_boxes'], is_training=True)
    model.load_state_dict(torch.load(args.weights)['state_dict'], strict=False)
    print('Load successfully from checkpoint: %s'%args.weights)
    model.eval()
    model.to(device)

    
    trainer = BaseTrainer(model, None, None, device, data_loader=test_loader, test_data_loader=test_loader, config=cfg)
    stats = trainer.test()
    print_s(stats, cfg['nc'])
    
    