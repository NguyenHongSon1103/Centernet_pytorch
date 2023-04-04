import os
import cv2
from torch.utils.data import Dataset
import math
import numpy as np
# import sys
# sys.path.append('/data2/sonnh/E2EObjectDetection/Centernet')
from augmenter import VisualAugmenter, MiscAugmenter, AdvancedAugmenter
from assigner import Assigner

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
    boxes = boxes.astype('float32')
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

class Generator(Dataset):
    def __init__(self, data, hparams, mode='train'):
        """
        Initialize Generator object.

        Args:
            data: dictionary with 2 keys: 'im_path' and 'lb_path'
            hparams: a config dictionary
        """
        self.data = data
        self.resizer = resize_methods['keep']
        self.batch_size = hparams['batch_size']
        self.input_size = hparams['input_size']
        self.stride = 4
        self.output_size = self.input_size // self.stride
        self.max_objects = hparams['max_objects']
        self.num_classes = hparams['num_classes']
        self.mode = mode

        ''' Old method
        self.visual_augmenter = VisualEffect(color_prob=0.25, contrast_prob=0.25,
            brightness_prob=0.25, sharpness_prob=0.25, autocontrast_prob=0.25,
            equalize_prob=0.25, solarize_prob=0.25)
        self.misc_augmenter = MiscEffect(multi_scale_prob=0.15, rotate_prob=0.15, flip_prob=0.15, crop_prob=0.15, translate_prob=0.15)
        '''
        ## New aumgenter
        self.visual_augmenter = VisualAugmenter(keep_prob=0.5)
        self.misc_augmenter = MiscAugmenter(keep_prob=0.5)
        self.advanced_augmenter = AdvancedAugmenter(keep_prob=0.5)

    def on_epoch_end(self):
        np.random.shuffle(self.data)

    def __len__(self):
        """
        Number of batches for generator.
        """
        nums = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size == 0:
            return nums
        return nums + 1

    def get_group(self, batch):
        """
        Abstract function, need to implement
        """
        pass

    def sample_batch(self, idx):
        """
        Abstract function, need to implement
        """
        pass

    def __getitem__(self, idx):
        batch = self.sample_batch(idx)
        group_images, group_boxes, group_ids = self.get_group(batch)

        ##Augmentation
        if self.mode == 'train':
            group_images_aug, group_boxes_aug, group_ids_aug = [], [], []
            for image, boxes, class_id in zip(group_images, group_boxes, group_ids):
                if len(group_images) > 4:
                    idxs = np.random.choice(np.arange(len(group_images)), 4)
                    list_data = [{'image':group_images[i], 'boxes':group_boxes[i], 'class_ids':group_ids[i]} for i in idxs]
                    item = self.advanced_augmenter(list_data)
                else:
                    item = {'image':image, 'boxes':boxes, 'class_ids':class_id}

                item = self.visual_augmenter(item)
                item = self.misc_augmenter(item)

                group_images_aug.append(item['image'])
                group_boxes_aug.append(item['boxes'])
                group_ids_aug.append(item['class_ids'])
            group_images = group_images_aug
            group_boxes = group_boxes_aug
            group_ids = group_ids_aug

        images, batch_hm, batch_wh, batch_reg = [], [], [], []
        for image, boxes, class_id in zip(group_images, group_boxes, group_ids):
            image, boxes = self.resizer(self.input_size, image, boxes) #512x512
            image = self.preprocess_image(image)
#             print(boxes)
            h, w = image.shape[:2]
            hm, wh, reg = self.compute_targets_each_image(boxes, class_id)
#             print(boxes, np.sum(hm), np.sum(wh), np.sum(reg), np.where(hm == 1.0))
            images.append(image)
            batch_hm.append(hm)
            batch_wh.append(wh)
            batch_reg.append(reg)

        outputs = np.concatenate([np.array(batch_wh, dtype=np.float32),
                                  np.array(batch_reg, dtype=np.float32),
                                  np.array(batch_hm, dtype=np.float32)
                                 ], -1)
        return np.array(images, dtype=np.float32), outputs

    def get_heatmap_per_box(self, heatmap, cls_id, ct_int, size):
        h, w = size
        radius = gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=0.7)
        radius = max(0, int(radius))
        heatmap[..., cls_id] = draw_gaussian_2(heatmap[...,cls_id], ct_int, radius)
        return heatmap

    def compute_targets_each_image(self, boxes, class_id):
        hm = np.zeros((self.output_size, self.output_size, self.num_classes), dtype=np.float32)
        whm = np.zeros((self.output_size, self.output_size, 2), dtype=np.float32)
        reg = np.zeros((self.output_size, self.output_size, 2), dtype=np.float32)
        for i, (box, cls_id) in enumerate(zip(boxes, class_id)):
            #scale box to output size
            xmin, ymin, xmax, ymax = [p/self.stride for p in box]
            h_box, w_box = ymax - ymin, xmax - xmin
            if h_box < 0 or w_box < 0:
                continue
            x_center, y_center = (xmax + xmin) / 2.0, (ymax + ymin) / 2.0
            x_center_floor, y_center_floor = int(np.floor(x_center)), int(np.floor(y_center))
            hm = self.get_heatmap_per_box(hm, cls_id, (x_center_floor, y_center_floor), (h_box, w_box))
            whm[x_center_floor, y_center_floor] = [w_box, h_box]
            reg[x_center_floor, y_center_floor] = x_center - x_center_floor, y_center - y_center_floor
#             print(ct - ct_int)
#         print(np.where(hm == 1))
#         print('------')
        return hm, whm, reg

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

#         image[..., 0] -= 103.939
#         image[..., 1] -= 116.779
#         image[..., 2] -= 123.68
        return image / 255.0

    def reverse_preprocess(self, image):
#         image[..., 0] += 103.939
#         image[..., 1] += 116.779
#         image[..., 2] += 123.68
        image *= 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_name2idx_mapper(self, path):
        '''
        from label name to index
        '''
        with open(path, 'r') as f:
            classes_dict = json.load(f)
        return classes_dict