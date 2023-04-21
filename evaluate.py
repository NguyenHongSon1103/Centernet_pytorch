from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import xml.etree.ElementTree as ET
import os
from utils import HiddenPrints

def generate_coco_format(dataset, save_path):
    class_names = dataset.mapper.keys()
    # for evaluation with pycocotools
    labels = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        labels["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    print('Check annotations ...')
    for i, item in enumerate(dataset.data):
        name = item['im_path'].split('/')[-1]
        img_id = os.path.splitext(name)[0]
        img_w, img_h = Image.open(item['im_path']).size
        labels["images"].append(
            {
                "file_name": name,
                "id": img_id,
                "width": img_w,
                "height": img_h,
            }
        )
        boxes, obj_names = item['boxes'], item['class_names']
        for box, obj_name in zip(boxes, obj_names):
            x1, y1, x2, y2 = [float(p) for p in box]
            # cls_id starts from 0
            cls_id = dataset.mapper[obj_name]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            labels["annotations"].append(
                {
                    "area": h * w,
                    "bbox": [x1, y1, w, h],
                    "category_id": cls_id,
                    "id": ann_id,
                    "image_id": img_id,
                    "iscrowd": 0,
                    # mask
                    "segmentation": [],
                }
            )
            ann_id += 1

    with open(save_path, "w") as f:
        json.dump(labels, f)
    print(f"Convert to COCO format finished. Resutls saved in {save_path}")

def generate_coco_format_predict(all_predictions, save_path):
    """
    Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        all_predictions: All predictions on validation set (May be harm if large val set ?)
        save_path: direction to save preditions json file
    """
    # start collecting results
    results = []
    for i, item in enumerate(all_predictions):
        boxes, scores, class_ids = item['boxes'], item['scores'], item['class_ids']
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
    
    