import json
import os
import cv2
from torch.utils.data import Dataset
import math
import numpy as np
from .augmenter import VisualAugmenter, MiscAugmenter, AdvancedAugmenter
from .assigner import Assigner
from copy import deepcopy
from .utils import parse_xml, check_is_image, Resizer
from tqdm import tqdm
from PIL import Image

def load_data(img_dirs):
    '''
    Default: each set contains 2 folder: images and annotations
    '''
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]

    image_paths = []
    label_paths = []
    for img_dir in img_dirs:
        lb_dir = img_dir.replace('images', 'annotations')
        
        for name in os.listdir(img_dir):
            if not check_is_image(name):
                continue
            image_paths.append(os.path.join(img_dir,  name))
            label_paths.append(os.path.join(lb_dir,  '.'.join(name.split('.')[:-1])+'.xml'))
    
    print('Scanning: ')
    background = 0
    data = []
    pbar = tqdm(enumerate(image_paths), total=len(image_paths))
    for i, fp in pbar:
        xp = label_paths[i]
        if os.path.exists(xp):
            boxes, class_names = parse_xml(xp)
            data.append({'im_path':fp, 'boxes':boxes, 'class_names':class_names})
        else:
            background += 1
        
        pbar.set_description(desc='Images: %d | Background: %d'%(len(data), background))
    
    return data

class Generator(Dataset):
    def __init__(self, hparams, mode='train'):
        """
        Centernet dataset

        Args:
            data: dictionary with 2 keys: 'im_path' and 'lb_path'
            hparams: a config dictionary
        """
        self.img_dirs = hparams[mode]
        self.batch_size = hparams['batch_size']
        self.input_size = hparams['input_size']
        self.resizer = Resizer(self.input_size, mode='letterbox')
        self.stride = 4
        self.output_size = self.input_size // self.stride
        self.max_objects = hparams['max_boxes']
        self.num_classes = hparams['nc']
        self.mode = mode
        self.mapper = hparams['names']

        self.data = load_data(self.img_dirs)
        self.assigner = Assigner(self.num_classes, self.input_size, self.stride, self.max_objects)

        ## New aumgenter
        self.visual_augmenter = VisualAugmenter(keep_prob=0.5)
        self.misc_augmenter = MiscAugmenter(keep_prob=0.5)
        self.advanced_augmenter = AdvancedAugmenter(keep_prob=0.5)
        
    def on_epoch_end(self):
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = deepcopy(self.data[idx])
        image = cv2.imread(d['im_path'])
        boxes, class_names = d['boxes'], d['class_names']
        class_ids = [self.mapper[name] for name in class_names]

        item = {'image':image, 'boxes':boxes, 'class_ids':class_ids}

        ##Augmentation
        if self.mode == 'train':
            item = self.visual_augmenter(item)
            item = self.misc_augmenter(item)

        item['image'], item['boxes'] = self.resizer(item['image'], item['boxes']) #640x640
        item['image'] = self.preprocess_image(item['image'])

        hm, wh, reg, indices = self.assigner(item['boxes'], item['class_ids'])
        if self.mode == 'val':
            return item['image'], (hm, wh, reg, indices), d['im_path']

        return item['image'], (hm, wh, reg, indices)

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        return image / 255.0

    def reverse_preprocess(self, image):
        image *= 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def generate_coco_format(self, save_path):
        class_names = self.mapper.keys()
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        print('Check annotations ...')
        for i, item in enumerate(self.data):
            name = item['im_path'].split('/')[-1]
            img_id = os.path.splitext(name)[0]
            img_w, img_h = Image.open(item['im_path']).size
            dataset["images"].append(
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
                cls_id = self.mapper[obj_name]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dataset["annotations"].append(
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
            json.dump(dataset, f)
        print(f"Convert to COCO format finished. Resutls saved in {save_path}")