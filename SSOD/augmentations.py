from copy import deepcopy
import math
import albumentations as A
import cv2
import numpy as np
import os

class WeakAug:
    def __init__(self, config):
        '''
        augmentation
        Note: use rotate in affine cause lost box, use safe rotate instead
        '''
        self.p = config['keep']
                
        affine = config['affine']
        rotate = config['rotate']
        
        T = [
            # A.BBoxSafeRandomCrop(0.0, p=config['crop']),
            A.RandomSizedBBoxSafeCrop(640, 640, 0, p=config['crop']),
            A.Affine(scale={'x':affine['scale_x'], 'y':affine['scale_y']},
                    keep_ratio=affine['keep_ratio'],
                    rotate=None, shear=affine['shear'], p=affine['prob']),
            A.SafeRotate (limit=rotate['degree'], border_mode=cv2.BORDER_CONSTANT,
                          value=0, p=rotate['prob']),
            A.HorizontalFlip(p=config['hflip']),
            A.VerticalFlip(p=config['vflip']),
        ]
    
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc'))
    
    def __call__(self, data):
        '''
        data: a dictionari with {'image': image, 'boxes': boxes, 'class_ids':class_ids}
        '''
        if np.random.random() < self.p:
            return data 
        ## Keep self.keep_prob original image
        boxes = [list(box) + [c] for box, c in zip(data['boxes'], data['class_ids'])]
        
        res = self.transform(image=data['image'], bboxes=boxes)
        augmented_data = deepcopy(data)
        augmented_data['image'] = res['image']
        augmented_data['boxes'] = [box[:4] for box in res['bboxes']]
        augmented_data['class_ids'] = [box[4] for box in res['bboxes']]
        return augmented_data

class StrongAug:
    def __init__(self, config):
        '''
        Augmentation methods used: ColorJitter, Grayscale, GaussianBlur, RandomCrop
        '''
        jitter = config['color_jitter']
        self.p = config['keep']
        self.transform = A.Compose([
             A.ColorJitter(brightness=jitter['brightness'], contrast=jitter['contrast'],
                          saturation=jitter['saturation'], hue=jitter['hue'], p=jitter['prob']),
            A.CoarseDropout (max_holes=3, max_height=50, max_width=50,
                                        min_height=10, min_width=10, p=0.2),
            # A.CoarseDropout (max_holes=1, max_height=100, max_width=100,
            #                             min_height=30, min_width=30, p=0.5),
            # A.CoarseDropout (max_holes=1, max_height=200, max_width=200,
            #                             min_height=50, min_width=50, p=0.3),
            A.GaussianBlur(p=config['blur']),
            # A.ImageCompression(quality_lower=75, quality_upper=100, p=config['image_compression']),
            A.ToGray(p=config['gray'])
        ])

    def __call__(self, image):
        if np.random.random() < self.p:
            return image 
        image = self.transform(image=image)['image']
        if len(image.shape) == 2:
            image = np.stack([image, image, image], -1)
        return image

class AdvancedAugmenter:
    def __init__(self, dataset, advanced_config, target_size=(640, 640)):
        self.p = advanced_config['keep'] #all prob
        self.target_size= target_size
        self.dataset = dataset #list
    
    def mosaic_v2(self, list_images, list_boxes):
        w, h = self.target_size
        scale_range = (0.3, 0.7)
        output_img = np.zeros([w, h, 3], dtype=np.uint8)
        scale_x = scale_range[0] + np.random.random() * (scale_range[1] - scale_range[0])
        scale_y = scale_range[0] + np.random.random() * (scale_range[1] - scale_range[0])
        divid_point_x = int(scale_x * w)
        divid_point_y = int(scale_y * h)

        new_anno = []
        for i in range(4):
            img = list_images[i]
            im_h, im_w = img.shape[:2]
            annos = [[xmin/im_w, ymin/im_h, xmax/im_w, ymax/im_h, c]
                for (xmin, ymin, xmax, ymax, c) in list_boxes[i]]
            if i == 0:  # top-left
                img = cv2.resize(img, (divid_point_x, divid_point_y))
                output_img[:divid_point_y, :divid_point_x, :] = img
                new_anno += [[xmin*scale_x, ymin*scale_y,
                             xmax*scale_x, ymax*scale_y, c]
                for (xmin, ymin, xmax, ymax, c) in annos]

            elif i == 1:  # top-right
                img = cv2.resize(img, (w - divid_point_x, divid_point_y))
                output_img[:divid_point_y, divid_point_x:w, :] = img
                new_anno += [[scale_x + xmin * (1 - scale_x), ymin*scale_y,
                             scale_x + xmax * (1 - scale_x), ymax*scale_y, c]
                for (xmin, ymin, xmax, ymax, c) in annos]

            elif i == 2:  # bottom-left
                img = cv2.resize(img, (divid_point_x, h - divid_point_y))
                output_img[divid_point_y:h, :divid_point_x, :] = img
                new_anno += [[xmin*scale_x, scale_y + ymin * (1 - scale_y),
                             xmax*scale_x, scale_y + ymax * (1 - scale_y), c]
                for (xmin, ymin, xmax, ymax, c) in annos]

            else:  # bottom-right
                img = cv2.resize(img, (w - divid_point_x, h - divid_point_y))
                output_img[divid_point_y:h, divid_point_x:w, :] = img
                new_anno += [[scale_x + xmin * (1 - scale_x), scale_y + ymin * (1 - scale_y),
                             scale_x + xmax * (1 - scale_x), scale_y + ymax * (1 - scale_y), c]
                for (xmin, ymin, xmax, ymax, c) in annos]

        #filter out anno with height or width smaller than 0.02 image size
        new_anno = [anno for anno in new_anno if
                    0.02 < (anno[2]-anno[0]) and 0.02 < (anno[3] - anno[1])]

        new_anno = [[xmin*w, ymin*h,
                             xmax*w, ymax*h, c]
                for (xmin, ymin, xmax, ymax, c) in new_anno]

        return output_img, new_anno

    def __call__(self, item):
        '''
        item: a dictionari with {'image': image, 'boxes': boxes, 'class_ids':class_ids}
        '''
        ## Keep self.keep_prob original image
        if np.random.random() < self.p:
            return item
        
        #Sample 3 more data
        list_data = [item]
        samples = np.random.choice(self.dataset.data, 3)
        for sample in samples:
            list_data.append(self.dataset.load_item(sample))
            
        list_images, list_boxes = [], []
        for data in list_data:
            list_images.append(data['image'])
            list_boxes.append([list(box) + [obj_name] for box, obj_name in zip(data['boxes'], data['class_ids'])])
        new_image, new_boxes = self.mosaic_v2(list_images, list_boxes)
        augmented_data = dict()
        augmented_data['image'] = new_image
        augmented_data['boxes'] = [box[:4] for box in new_boxes]
        augmented_data['class_ids'] = [box[4] for box in new_boxes]
        return augmented_data