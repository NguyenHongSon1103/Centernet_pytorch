import hashlib
import json
import os
import cv2
from torch.utils.data import Dataset
import math
import numpy as np
import sys
sys.path.append(os.getcwd())
from .augmenter import VisualAugmenter, SpatialAugmenter, AdvancedAugmenter
from .old_augmenter import OldTransformer
from .assigner import Assigner
from copy import deepcopy
from utils import parse_xml, check_is_image, Resizer, colorstr
from tqdm import tqdm
from PIL import Image
import logging

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

def load_unsup_data(img_dirs):
    '''
    Default: each set contains 2 folder: images and annotations
    '''
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]

    image_paths = []
    for img_dir in img_dirs:
        for name in tqdm(os.listdir(img_dir)):
            if not check_is_image(name):
                continue
            image_paths.append(os.path.join(img_dir,  name))
    return image_paths

'''
Train: 
    each batch contains: sup: (image_strong +label) and  (image_weak + label)
                         unsup: image_strong and image_weak
Val: image + label
'''

class UnsupGenerator(Dataset):
    def __init__(self, hparams, mode='train'):
        """
        Centernet dataset

        Args:
            data: dictionary with 2 keys: 'im_path' and 'lb_path'
            hparams: a config dictionary
        """
        self.batch_size = hparams['batch_size']
        self.input_size = hparams['input_size']
        self.resizer = Resizer(self.input_size, mode='keep')

        self.unsup_data = load_unsup_data(hparams['train_unsup'])
        
        self.weak_augmenter = WeakAug(multi_scale_prob=0.5, rotate_prob=0.05, flip_prob=0.5)
        self.strong_augmenter = StrongAug()
    
    def __len__(self):
        return len(self.unsup_data)
    
    def __getitem__(self, idx):
        unsup_img = cv2.imread(self.unsup_data[idx])
        
        unsup_weak_img = self.weak_augmenter(unsup_img, None)
        unsup_strong_img = self.strong_augmenter(unsup_weak_img)
        
        h, w = unsup_weak_img.shape[:2]
        unsup_weak_img = cv2.cvtColor(unsup_weak_img, cv2.COLOR_BGR2RGB)
        unsup_strong_img = cv2.cvtColor(unsup_strong_img, cv2.COLOR_BGR2RGB)
        
        unsup_weak_img = self.resizer.resize_image(unsup_weak_img).astype('float32') / 255.0
        unsup_strong_img = self.resizer.resize_image(unsup_strong_img).astype('float32') / 255.0

        return unsup_weak_img, unsup_strong_img

class Generator(Dataset):
    def __init__(self, hparams, mode='train'):
        """
        Centernet dataset

        Args:
            data: dictionary with 2 keys: 'im_path' and 'lb_path'
            hparams: a config dictionary
        """
        self.batch_size = hparams['batch_size']
        self.input_size = hparams['input_size']
        # self.resizer = Resizer(self.input_size, mode='letterbox')
        self.resizer = Resizer(self.input_size, mode='keep')
        self.stride = 4
        self.max_objects = hparams['max_boxes']
        self.num_classes = hparams['nc']
        self.mode = mode
        self.mapper = hparams['names']
        self.save_dir = os.path.join(hparams['save_dir'], 'preprocessed')
        os.makedirs(self.save_dir, exist_ok=True)

        self.data = load_data(hparams[mode], self.mode)
        
        self.assigner = Assigner(self.num_classes, self.input_size, self.stride, self.max_objects)

        self.weak_augmenter = WeakAug(multi_scale_prob=0.5, rotate_prob=0.05, flip_prob=0.5)
        self.strong_augmenter = StrongAug()
                        
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
        
        if self.mode != 'train':
            h, w = item['image'].shape[:2]
            item['image'] = cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB)
            item['image'] = self.resizer.resize_image(item['image'])
            item['boxes'] = self.resizer.resize_boxes(item['boxes'], (h, w)) #640x640

            item['image'] = item['image'].astype('float32') / 255.0

            hm, wh, reg, indices = self.assigner(item['boxes'], item['class_ids'])

            return item['image'], (hm, wh, reg, indices), d['im_path']
        
        sup_weak_img, boxes = self.weak_augmenter(item['image'], item['boxes'])
        sup_strong_img = self.strong_augmenter(sup_weak_img)
        
        h, w = sup_weak_img.shape[:2]
        sup_weak_img = cv2.cvtColor(sup_weak_img, cv2.COLOR_BGR2RGB)
        sup_strong_img = cv2.cvtColor(sup_strong_img, cv2.COLOR_BGR2RGB)
        
        sup_weak_img = self.resizer.resize_image(sup_weak_img).astype('float32') / 255.0
        sup_strong_img = self.resizer.resize_image(sup_strong_img).astype('float32') / 255.0
        
        boxes = self.resizer.resize_boxes(boxes, (h, w)) #640x640

        hm, wh, reg, indices = self.assigner(boxes, item['class_ids'])

        return sup_weak_img, sup_strong_img, (hm, wh, reg, indices), d['im_path']
