import os
import cv2
from torch.utils.data import Dataset
import math
import numpy as np
from .augmenter import VisualAugmenter, MiscAugmenter, AdvancedAugmenter
from .assigner import Assigner
from copy import deepcopy
from .utils import parse_xml, check_is_image
from tqdm import tqdm

def letterbox(size, im, boxes, color=(114, 114, 114), stride=32):
    ## auto = True, scaleup = False
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(size, int):
        new_shape = (size, size)
    else: new_shape = size

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # w, h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    boxes = np.array(boxes).astype('float32')
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        boxes[..., [0, 2]] *= new_unpad[0]/shape[1]
        boxes[..., [1, 3]] *= new_unpad[1]/shape[0]
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    boxes[..., [0, 2]] += dw
    boxes[..., [1, 3]] += dh
    return im, boxes.astype('int32')

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
        self.resizer = letterbox
        self.batch_size = hparams['batch_size']
        self.input_size = hparams['input_size']
        self.stride = 4
        self.output_size = self.input_size // self.stride
        self.max_objects = hparams['max_objects']
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

        item['image'], item['boxes'] = self.resizer(self.input_size, item['image'], item['boxes']) #512x512
        item['image'] = self.preprocess_image(item['image'])

        hm, wh, reg, indices = self.assigner(item['boxes'], item['class_ids'])

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
