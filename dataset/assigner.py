import math
import numpy as np
from .utils import gaussian_radius, draw_gaussian, draw_gaussian_2, draw_msra_gaussian

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

        return hm, whm, reg
    
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