import hashlib
import json
import os
import cv2
from torch.utils.data import Dataset
import math
import numpy as np
import sys
sys.path.append(os.getcwd())
from .augmenter import Augmenter, AdvancedAugmenter
from .assigner import Assigner
from copy import deepcopy
from utils import parse_xml, check_is_image, Resizer, colorstr
from tqdm import tqdm
from PIL import Image
import logging
from time import time

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash
    
def load_data(img_dirs, mode='train'):
    '''
    Default: each set contains 2 folder: images and annotations
    '''
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]

    label_dirs = [img_dir.replace('images', 'annotations') for img_dir in img_dirs]

    image_paths = []
    label_paths = []
    for img_dir, lb_dir in zip(img_dirs, label_dirs):
        
        for name in os.listdir(img_dir):
            if not check_is_image(name):
                continue
            image_paths.append(os.path.join(img_dir,  name))
            label_paths.append(os.path.join(lb_dir,  '.'.join(name.split('.')[:-1])+'.xml'))
    
    LOGGER.info(colorstr('blue', mode.upper()))
    # Load from cache
    cache_path = label_dirs[0] +'.cache'
    
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True).item()  # load dict
        LOGGER.info('Loading labels from cache ...' )
        if cache['hash'] == get_hash(label_dirs + img_dirs):  # identical hash
            nf, nm, ne, _ = cache['results']
            data = cache['data']
            LOGGER.info('Total: %d | Images: %d | Background: %d | Empty: %d'%(len(data), nf, nm, ne))
            return data
        else:
            LOGGER.error('Cannot load from cache, try to load from source paths')
    # If cannot load from cache, load from folder
    nm, nf, ne = 0, 0, 0  # number missing, found, empty
    data = []
    pbar = tqdm(enumerate(image_paths), total=len(image_paths))
    for i, fp in pbar:
        xp = label_paths[i]
        if not os.path.exists(xp):
            nm += 1
            continue
        boxes, class_names = parse_xml(xp)
        if len(boxes) == 0:
            ne += 1
            continue
        data.append({'im_path':fp, 'boxes':boxes, 'class_names':class_names})
        nf += 1
        
        pbar.set_description(desc='Total: %d | Images: %d | Background: %d | Empty: %d'%(len(data), nf, nm, ne))
    
    # Cache file: 
    if nf == 0:
        LOGGER.warning('WARNING ⚠️ No labels found')
    x = {}
    x['data'] = data
    x['hash'] = get_hash(label_dirs + img_dirs)
    x['results'] = nf, nm, ne, len(image_paths)
    np.save(cache_path, x)  # save cache for next time
    os.rename(cache_path+'.npy', cache_path) # remove .npy suffix
    LOGGER.info(f'New cache created: {cache_path}')
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
        self.resizer = Resizer(self.input_size, mode='keep')
        self.stride = 4
        self.max_objects = hparams['max_boxes']
        self.num_classes = hparams['nc']
        self.mode = mode
        self.mapper = hparams['names']
        self.save_dir = os.path.join(hparams['save_dir'], 'preprocessed')
        os.makedirs(self.save_dir, exist_ok=True)

        self.data = load_data(self.img_dirs, self.mode)
        # np.random.seed(12345)
        self.on_epoch_end()
        # if mode == 'train':
        #     split_idx = int(len(self.data) * 0.10) #1%, 5%, 10%
        #     self.data = self.data[:split_idx]
            
        self.assigner = Assigner(self.num_classes, self.input_size, self.stride, self.max_objects)

        ## New aumgenter
        self.augmenter = Augmenter(hparams['augment'])
        self.advanced_augmenter = AdvancedAugmenter(self, hparams['advanced'])
                        
    def on_epoch_end(self):
        np.random.shuffle(self.data)
    
    def load_item(self, d):
        image = cv2.imread(d['im_path'])
        h, w = image.shape[:2]
        boxes, class_names = d['boxes'], d['class_names']
        boxes = np.array(boxes)
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, w)
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, h)
        class_ids = np.array([self.mapper[name] for name in class_names])

        item = {'image':image, 'boxes':boxes, 'class_ids':class_ids}
        return item
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        item = self.load_item(d)
        ##Augmentation
        if self.mode == 'train':
            try:
                src_item = deepcopy(item)
                # if self.advanced_augmenter is not None:
                src_item = self.advanced_augmenter(src_item)
                src_item = self.augmenter(src_item) 
                #bug: Lost box after spatial transform, happen a few time
                #still not know which transformation caused this
                if len(src_item['boxes']) > 0:
                    item = src_item
            except:
                pass
                
        h, w = item['image'].shape[:2]
        item['image'] = cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB)
        item['image'] = self.resizer.resize_image(item['image'])
        item['boxes'] = self.resizer.resize_boxes(item['boxes'], (h, w)) #640x640
        
        # item['image'] = item['image'].astype('float32') - [122.67891434, 116.66876762, 104.00698793]
        item['image'] = item['image'].astype('float32') / 255.0
    
        hm, wh, reg, indices = self.assigner(item['boxes'], item['class_ids'])

        return item['image'], (hm, wh, reg, indices), d['im_path']

    def reverse_preprocess(self, image):
        image *= 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
