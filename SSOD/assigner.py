import math
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

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

class Assigner:
    def __init__(self, num_classes, input_size=(640, 640), stride=4, max_object=100):
        """
        Class for convert boxes and class ids to labels 
        -> heatmap, whmap and regmap
        """
        self.input_size = input_size
        self.stride = stride
        self.output_size = self.input_size // self.stride
        self.max_objects = max_object
        self.num_classes = num_classes

    def __call__(self, boxes ,class_ids):

        hm, wh, reg, indices = self.compute_target(boxes, class_ids)
        return hm, wh, reg, indices

    def get_heatmap_per_box(self, heatmap, cls_id, ct_int, size):
        h, w = size
        radius = gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=0.7)
        radius = max(0, int(radius))
        heatmap[..., cls_id] = draw_gaussian_2(heatmap[...,cls_id], ct_int, radius)
        return heatmap

#     def compute_targets_each_image(self, boxes, class_id):
#         hm = np.zeros((self.output_size, self.output_size, self.num_classes), dtype=np.float32)
#         whm = np.zeros((self.output_size, self.output_size, 2), dtype=np.float32)
#         reg = np.zeros((self.output_size, self.output_size, 2), dtype=np.float32)
#         for i, (box, cls_id) in enumerate(zip(boxes, class_id)):
#             #scale box to output size
#             xmin, ymin, xmax, ymax = [p/self.stride for p in box]
#             h_box, w_box = ymax - ymin, xmax - xmin
#             if h_box < 0 or w_box < 0:
#                 continue
#             x_center, y_center = (xmax + xmin) / 2.0, (ymax + ymin) / 2.0
#             x_center_floor, y_center_floor = int(np.floor(x_center)), int(np.floor(y_center))
#             hm = self.get_heatmap_per_box(hm, cls_id, (x_center_floor, y_center_floor), (h_box, w_box))
#             whm[x_center_floor, y_center_floor] = [w_box, h_box]
#             reg[x_center_floor, y_center_floor] = x_center - x_center_floor, y_center - y_center_floor

#         return hm, whm, reg
    
    def compute_target(self, boxes, class_id):
        hm = np.zeros((self.output_size, self.output_size, self.num_classes), dtype=np.float32)
        whm = np.zeros((self.max_objects, 2), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices = np.zeros((self.max_objects), dtype=np.float32)

        for i, (box, cls_id) in enumerate(zip(boxes, class_id)):
            #scale box to output size
            xmin, ymin, xmax, ymax = [p/self.stride for p in box]
            h_box, w_box = ymax - ymin, xmax - xmin
            if h_box < 0 or w_box < 0:
                continue
            x_center, y_center = (xmax + xmin) / 2.0, (ymax + ymin) / 2.0
            x_center_floor, y_center_floor = int(np.floor(x_center)), int(np.floor(y_center))
            hm = self.get_heatmap_per_box(hm, cls_id, (x_center_floor, y_center_floor), (h_box, w_box))
            whm[i] = [w_box, h_box]
            reg[i] = x_center - x_center_floor, y_center - y_center_floor
            indices[i] = y_center_floor * self.output_size + x_center_floor

        return hm, whm, reg, indices