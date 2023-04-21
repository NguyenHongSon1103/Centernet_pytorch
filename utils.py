import numpy as np
import xml.etree.ElementTree as ET
import json
import cv2
import sys
import cv2
import numpy as np
import os
import pandas as pd

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Resizer:
    def __init__(self, size=(640, 640), mode='letterbox'):
        self.size = (size, size) if isinstance(size, int) else size
        assert mode in ['letterbox', 'keep', 'no keep'], "Unknown mode: %s"%mode
        self.mode = mode
    
    def resize_image(self, image):
        if self.mode == 'letterbox':
            new_img = self.letterbox(image, (114, 114, 114))
        elif self.mode == 'keep':
            new_img = self.resize_keep_ar(image)
        elif self.mode == 'no keep':
            new_img = self.resize_wo_keep_ar(image)
            
        return new_img
    
    def resize_boxes(self, original_boxes, original_shape):
        '''
        Convert boxes from original image size back to preprocessed image size
        '''
        
        h, w = original_shape[:2]
        scale_w = self.size[0] / w
        scale_h = self.size[1] / h
        
        boxes = np.copy(original_boxes).astype('float32')
        if self.mode == 'letterbox':
            r = min(scale_w, scale_h)
            r = min(r, 1.0)
            new_unpad = int(round(w * r)), int(round(h * r)) # w, h
            dw, dh = self.size[0] - new_unpad[0], self.size[1] - new_unpad[1]  # wh padding

            dw /= 2  # divide padding into 2 sides
            dh /= 2
            if [w, h] != new_unpad:
                boxes[..., [0, 2]] *= new_unpad[0]/w
                boxes[..., [1, 3]] *= new_unpad[1]/h
            boxes[..., [0, 2]] += dw
            boxes[..., [1, 3]] += dh
                        
        elif self.mode == 'keep':
            scale = min(scale_w, scale_h)
            boxes *= scale

        elif self.mode == 'no keep':
            boxes[..., [0, 2]] *= scale_w
            boxes[..., [1, 3]] *= scale_h
            
        return boxes
    
    def rescale_boxes(self, boxes, original_shape):
        '''
        Convert boxes from preprocessed image size back to original size
        '''
        
        h, w = original_shape[:2]
        scale_w = self.size[0] / w
        scale_h = self.size[1] / h
        boxes = np.array(boxes)
        if self.mode == 'letterbox':
            # Rescale boxes (xyxy) from img1_shape to img0_shape
            gain = min(scale_w, scale_h)  # gain  = old / new
            pad = (self.size[0] - w * gain) / 2, (self.size[1] - h * gain) / 2  # wh padding

            boxes[..., [0, 2]] -= pad[0]  # x padding
            boxes[..., [1, 3]] -= pad[1]  # y padding
            boxes[..., :4] /= gain
            # clip_boxes(boxes, img0_shape)
            
        elif self.mode == 'keep':
            scale = min(scale_w, scale_h)
            boxes = boxes / scale

        elif self.mode == 'no keep':
            '''Not implement'''
            pass

        return boxes

    def resize_keep_ar(self, image):
        h, w, c = image.shape
        scale_w = self.size[0] / w
        scale_h = self.size[1] / h
        scale = min(scale_w, scale_h)
        h = int(h * scale)
        w = int(w * scale)
        padimg = np.zeros((self.size[0], self.size[1], c), image.dtype)
        padimg[:h, :w] = cv2.resize(image, (w, h))
        return padimg

    def resize_wo_keep_ar(self, image):
        h, w, c = image.shape
        resized = cv2.resize(image, self.size)
        return resized
    
    def letterbox(self, im, color=(114, 114, 114), stride=32):
        ## auto = True, scaleup = False
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        # if isinstance(size, int):
        #     new_shape = (size, size)
        # else: new_shape = size
        new_shape = [self.size[0], self.size[1]]
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # w, h
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

def check_is_image(name:str):
    return name.endswith(('.jpg', '.png', '.jpeg'))

def parse_coco_json(path):
    with open(path, 'r') as f:
        labels = json.load(f)
    images, annotations, categories = labels['images'], labels['annotations'], labels['categories']
    classes_dict = {c['id']:c['name'] for c in categories}

    # Get map boxes to image
    annotations_dict = {image['id']:{'filename':image['file_name'],
                                     'boxes':[], 'class_id':[]} for image in images}
    for ano in annotations:
        annotations_dict[ano['image_id']]['boxes'].append(ano['bbox'])
        annotations_dict[ano['image_id']]['class_id'].append(ano['category_id'])
    
    data = list(annotations_dict.values())
    return data, classes_dict

def parse_xml(xml):
    root = ET.parse(xml).getroot()
    objs = root.findall('object')
    boxes, ymins, obj_names = [], [], []
    for obj in objs:
        obj_name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        ymins.append(ymin)
        boxes.append([xmin, ymin, xmax, ymax])
        obj_names.append(obj_name)
    indices = np.argsort(ymins)
    boxes = [boxes[i] for i in indices]
    obj_names = [obj_names[i] for i in indices]
    return boxes, obj_names

def emojis(string=''):
    # Return platform-dependent emoji-safe version of string
    return string.encode().decode('ascii', 'ignore') #if WINDOWS else string

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def save_csv(info, save_dir, header=True):
    assert isinstance(info, dict), "Save only dictionary"
    df = pd.DataFrame(info, index=[0])
    df.to_csv(os.path.join(save_dir, 'results.csv'), header=header,
              index=False, index_label=False, mode='a+')

def save_batch(impaths, images, targets, blend_heatmap=True, size=640, save_dir='', name=''):
    drews = []
    for i, (impath, src_img) in enumerate(zip(impaths, images)):
        imname = os.path.basename(impath)
        img = src_img.copy()
        #convert img from float to uint8
        img = (img*255.0).astype('uint8')
        # Convert target from label to boxes
        hm, wh, reg, indices = [targets[idx][i] for idx in range(4)]
        cls_ids = np.where(hm == 1)[-1]
        true_indices = indices[indices > 0]
        true_wh, true_reg = wh[indices > 0], reg[indices > 0]
        xc = true_indices % (size//4)
        yc = true_indices // (size//4)
        xmin = xc + true_reg[:, 0] - true_wh[:, 0]/2
        # print(xc.shape, yc.shape, true_reg.shape, true_wh.shape)
        boxes = np.stack([
            xc + true_reg[:, 0] - true_wh[:, 0]/2,
            yc + true_reg[:, 1] - true_wh[:, 1]/2,
            xc + true_reg[:, 0] + true_wh[:, 0]/2,
            yc + true_reg[:, 1] + true_wh[:, 1]/2
        ], 0).transpose()
        boxes = np.array(boxes)*4 #160 to 640

        for box, cls_id in zip(boxes, cls_ids):
            x1, y1, x2, y2 = [int(p) for p in box]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            ret, baseline = cv2.getTextSize(str(cls_id), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            img = cv2.rectangle(img, (x1, y1- ret[1] - baseline), (x1 + ret[0], y1), (255, 255, 255), -1)
            img = cv2.putText(img, str(cls_id), (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            img = cv2.putText(img, imname, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        for cls_id in range(hm.shape[-1]):
            heat = cv2.applyColorMap((hm[..., cls_id]*255).astype('uint8'), cv2.COLORMAP_JET)
            heat = cv2.resize(heat, (size, size))
            img = cv2.addWeighted(img, 0.8, heat.astype('uint8'), 0.2, 0.0)
        
        drews.append(img)
        # Blend heatmap

    #merge each 9 images: 
    image = np.zeros((size*3, size*3, 3), dtype='uint8')

    for i in range(3):
        for j in range(3):
            if i*3+j > len(drews): break
            sc, ec, sr, er = i*size, (i+1)*size, j*size, (j+1)*size 

            image[sc:ec, sr:er] = drews[i*3+j]
    cv2.imwrite(os.path.join(save_dir, 'preprocessed', name), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    gaussian2D((3, 3))
