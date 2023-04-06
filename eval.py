from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import xml.etree.ElementTree as ET
import os
from dataset.utils import HiddenPrints

def evaluate(anno_json, pred_json):
    # load results in COCO evaluation tool
    coco_true = COCO(anno_json)
    coco_pred = coco_true.loadRes(pred_json)

    # run COCO evaluation
    with HiddenPrints():
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    #     coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval.stats

def evaluate_all(anno_json, pred_json):
    # load results in COCO evaluation tool
    coco_true = COCO(anno_json)
    coco_pred = coco_true.loadRes(pred_json)

    # run COCO evaluation
    stats = []
    for catId in coco_true.getCatIds():
        with HiddenPrints():
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
    
    
    