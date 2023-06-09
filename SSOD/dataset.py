import hashlib
import json
import os
import cv2
from torch.utils.data import Dataset
import math
import numpy as np
import sys
sys.path.append(os.getcwd())
from .augmentations import WeakAug, StrongAug, AdvancedAugmenter
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

class Generator(Dataset):
    def __init__(self, hparams, mode='train'):
        """
        Centernet dataset

        Args:
            data: dictionary with 2 keys: 'im_path' and 'lb_path'
            hparams: a config dictionary
        """
        self.batch_size = hparams['batch_size']
        self.unsup_batch_size = hparams['unsup_batch_size']
        self.input_size = hparams['input_size']
        self.resizer = Resizer(self.input_size, mode='keep')
        self.stride = 4
        self.max_objects = hparams['max_boxes']
        self.num_classes = hparams['nc']
        self.mode = mode
        self.mapper = hparams['names']
        self.save_dir = os.path.join(hparams['save_dir'], 'preprocessed')
        os.makedirs(self.save_dir, exist_ok=True)

        self.data = load_data(hparams[mode], self.mode)
            
        if self.mode == 'train':
            np.random.seed(12345)
            np.random.shuffle(self.data)
            split_idx = int(len(self.data) * float(hparams['partial'])) #1%, 5%, 10%
            self.unsup_data = self.data[split_idx:]
            self.data = self.data[:split_idx]
            
            # self.unsup_data = load_unsup_data(hparams['train_unsup'])
        
        self.assigner = Assigner(self.num_classes, self.input_size, self.stride, self.max_objects)

        self.weak_augmenter = WeakAug(hparams['augment'])
        self.strong_augmenter = StrongAug(hparams['augment'])
        self.advanced_augmenter = AdvancedAugmenter(self, hparams['advanced'])
        
        #disable random crop, auto apply augment
        hparams['augment']['keep'] = 0.0
        hparams['augment']['crop'] = 0.0
        self.unsup_weak_augmenter = WeakAug(hparams['augment'])
        
        hparams['augment']['color_jitter']['prob'] = 0.8
        hparams['augment']['blur'] = 0.5
        hparams['augment']['gray'] = 0.2
        self.unsup_strong_augmenter = StrongAug(hparams['augment'])
                        
    def on_epoch_end(self):
        np.random.shuffle(self.data)
        if self.mode == 'train':
            np.random.shuffle(self.unsup_data)
    
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
        if self.mode == 'train':
            return len(self.unsup_data)
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx%len(self.data)]
        item = self.load_item(d)
        
        if self.mode != 'train':
            h, w = item['image'].shape[:2]
            item['image'] = cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB)
            item['image'] = self.resizer.resize_image(item['image'])
            item['boxes'] = self.resizer.resize_boxes(item['boxes'], (h, w)) #640x640

            item['image'] = item['image'].astype('float32') / 255.0

            # hm, wh, reg, indices = self.assigner(item['boxes'], item['class_ids'])

            return item['image'], d['im_path']
        
        #Supervised
        item = self.advanced_augmenter(item)
        item = self.weak_augmenter(item)
        sup_weak_img, boxes, class_ids = item['image'], item['boxes'], item['class_ids']
        sup_strong_img = self.strong_augmenter(sup_weak_img)
        
        h, w = sup_strong_img.shape[:2]
        # sup_weak_img = cv2.cvtColor(sup_weak_img, cv2.COLOR_BGR2RGB)
        sup_strong_img = cv2.cvtColor(sup_strong_img, cv2.COLOR_BGR2RGB)
        
        # sup_weak_img = self.resizer.resize_image(sup_weak_img).astype('float32') / 255.0
        sup_strong_img = self.resizer.resize_image(sup_strong_img).astype('float32') / 255.0
        
        boxes = self.resizer.resize_boxes(boxes, (h, w)) #640x640

        hm, wh, reg, indices = self.assigner(boxes, item['class_ids'])
        
        #Unsupervised
        unsup_weak_images, unsup_strong_images = [], []
        for i in range(self.unsup_batch_size//self.batch_size):
            sample = np.random.choice(self.unsup_data)
            unsup_img = cv2.imread(sample['im_path'])

            weak_item = {'image':unsup_img, 'boxes':[], 'class_ids':[]}
            unsup_weak_img = self.unsup_weak_augmenter(weak_item)['image']
            unsup_strong_img = self.unsup_strong_augmenter(unsup_weak_img.copy())

            h, w = unsup_weak_img.shape[:2]
            unsup_weak_img = cv2.cvtColor(unsup_weak_img, cv2.COLOR_BGR2RGB)
            unsup_strong_img = cv2.cvtColor(unsup_strong_img, cv2.COLOR_BGR2RGB)

            unsup_weak_img = self.resizer.resize_image(unsup_weak_img).astype('float32') / 255.0
            unsup_strong_img = self.resizer.resize_image(unsup_strong_img).astype('float32') / 255.0
            
            unsup_weak_images.append(unsup_weak_img)
            unsup_strong_images.append(unsup_strong_img)

        return unsup_weak_images, unsup_strong_images, sup_strong_img, (hm, wh, reg, indices)
