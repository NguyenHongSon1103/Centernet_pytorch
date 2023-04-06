import numpy as np
import xml.etree.ElementTree as ET
import json
import cv2
import sys
import cv2
import numpy as np

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Resizer:
    def __init__(self, size=(640, 640), mode='letterbox') -> None:
        self.size = (size, size) if isinstance(size, int) else self.size
        self.mode = mode
    
    def __call__(self, image, boxes):
        if self.mode == 'letterbox':
            new_img, new_boxes = self.letterbox(image, boxes)
        elif self.mode == 'keep':
            new_img, new_boxes = self.resize_keep_ar(image, boxes)
        elif self.mode == 'no keep':
            new_img, new_boxes = self.resize_wo_keep_ar(image, boxes)
        else:
            print('method %s not support yet'%self.mode)
            assert False
        return new_img, new_boxes
    
    def rescale_boxes(self, boxes, original_shape):
        'img1_shape, boxes, img0_shape'
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
        else:
            print('method %s not support yet'%self.mode)
            assert False

        return boxes

    def resize_keep_ar(self, image, boxes):
        h, w, c = image.shape
        scale_w = self.size[0] / w
        scale_h = self.size[1] / h
        scale = min(scale_w, scale_h)
        h = int(h * scale)
        w = int(w * scale)
        padimg = np.zeros((self.size[0], self.size[1], c), image.dtype)
        padimg[:h, :w] = cv2.resize(image, (w, h))
        new_anns = []
        for box in boxes:
            box = np.array(box).astype(np.float32)
            box *= scale
            new_anns.append(box)
        return padimg, new_anns

    def resize_wo_keep_ar(self, image, boxes):
        h, w, c = image.shape
        resized = cv2.resize(image, self.size)
        scale_h, scale_w = self.size[1]/h, self.size[0]/w
        new_anns = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xmin, xmax = xmin * scale_w, xmax * scale_w
            ymin, ymax = ymin * scale_h, ymax * scale_h
            new_anns.append([xmin, ymin, xmax, ymax])
        return resized, new_anns
    
    def letterbox(self, im, boxes, color=(114, 114, 114), stride=32):
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

def draw_gaussian(heatmap, center, radius_h, radius_w, k=1):
    diameter_h = 2 * radius_h + 1
    diameter_w = 2 * radius_w + 1
    gaussian = gaussian2D((diameter_h, diameter_w), sigma_w=diameter_w / 6, sigma_h=diameter_h / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_w), min(width - x, radius_w + 1)
    top, bottom = min(y, radius_h), min(height - y, radius_h + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_h - top:radius_h + bottom, radius_w - left:radius_w + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def draw_gaussian_2(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D_2((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
#     print(heatmap.shape, masked_heatmap.shape, masked_gaussian.shape)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(masked_heatmap, masked_gaussian * k)

#     masked_heatmap = heatmap[x - left:x + right, y - top:y + bottom]
#     masked_gaussian = gaussian[radius - left:radius + right, radius - top:radius + bottom]
#     if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
#         heatmap[ x - left:x + right, y - top:y + bottom] = np.maximum(masked_heatmap, masked_gaussian * k)

    return heatmap


def gaussian2D(shape, sigma_w=1, sigma_h=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-((x * x) / (2 * sigma_w * sigma_w) + (y * y) / (2 * sigma_h * sigma_h)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian2D_2(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return max(0, min(r1, r2, r3))

if __name__ == '__main__':
    gaussian2D((3, 3))
