from copy import deepcopy
import math
import albumentations as A
import cv2
import numpy as np
import os

class VisualAugmenter:
    def __init__(self, keep_prob=0.3):
        self.p = keep_prob
        T = [
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.CLAHE(p=0.2),
            A.OneOf([
                A.Blur(p=0.3),
                A.GaussianBlur(p=0.4),
                A.MedianBlur(p=0.3)
            ], p=0.1),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.1),
            A.ToGray(p=0.1)
        ]
        self.transform = A.Compose(T, p=1.0)
    
    def __call__(self, data):
        '''
        data: a dictionari with {'image': image, 'boxes': boxes, 'class_ids':class_ids}
        '''
        ## Keep self.keep_prob original image
        if np.random.random() < self.p:
            return data
    
        res = self.transform(image=data['image'])['image']
        augmented_data = deepcopy(data)
        augmented_data['image'] = res
        return augmented_data

class MiscAugmenter:
    def __init__(self, keep_prob=0.3):
        self.p = keep_prob
        T = [
            A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.2),
            A.Affine(scale={'x':(0.5, 1.5), 'y':(0.5, 1.5)}, keep_ratio=False,
                    translate_percent=None,
                    rotate=(0, 5), shear=None, p=0.2),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2)
        ]
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc'))
    
    def __call__(self, data):
        '''
        data: a dictionari with {'image': image, 'boxes': boxes, 'class_ids':class_ids}
        '''
        ## Keep self.keep_prob original image
        if np.random.random() < self.p:
            return data
        boxes = [list(box) + [c] for box, c in zip(data['boxes'], data['class_ids'])]
        
        res = self.transform(image=data['image'], bboxes=boxes)
        augmented_data = deepcopy(data)
        augmented_data['image'] = res['image']
        augmented_data['boxes'] = [box[:4] for box in res['bboxes']]
        augmented_data['class_ids'] = [box[4] for box in res['bboxes']]
        return augmented_data

def resize(image, boxes, size):
    w, h = size
    ih, iw = image.shape[:2]
    scale_w, scale_h = w/iw, h/ih
    new_image = cv2.resize(image, (w, h))
    new_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        xmin, xmax = int(xmin*scale_w), int(xmax*scale_w)
        ymin, ymax = int(ymin*scale_h), int(ymax*scale_h)
        new_boxes.append([xmin, ymin, xmax, ymax, box[4]])
    return new_image, new_boxes

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[:4]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class AdvancedAugmenter:
    def __init__(self, keep_prob=0.7, target_size=(640, 640)):
        self.p = keep_prob
        self.target_size= target_size

    def mosaic_augmenter(self, list_images, list_boxes):
        '''
        list_images: 4 images
        list_boxes : 4 coressponding boxes, each box -> (x1, y1, x2, y2, c)
        target_size: w, h
        '''
        data = [resize(image, boxes, self.target_size) for image, boxes in zip(list_images, list_boxes)]
        images = [d[0] for d in data]
        box_datas = [d[1] for d in data]
        w, h = self.target_size
        min_offset_x = rand(0.3, 0.7)
        min_offset_y = rand(0.3, 0.7)
        
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros((w, h, 3))
        new_image[:cuty, :cutx, :] = images[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = images[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = images[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = images[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        new_boxes = merge_bboxes(box_datas, cutx, cuty)
        return new_image, new_boxes
    
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

        #filter out anno with height or width smaller than 0.02% image size
        new_anno = [anno for anno in new_anno if
                    0.02 < (anno[2]-anno[0]) and 0.02 < (anno[3] - anno[1])]

        new_anno = [[xmin*w, ymin*h,
                             xmax*w, ymax*h, c]
                for (xmin, ymin, xmax, ymax, c) in new_anno]

        return output_img, new_anno

    def __call__(self, list_data):
        '''
        list_data: list of 4 dictionari with {'image': image, 'boxes': boxes, 'class_ids':class_ids}
        '''
        ## Keep self.keep_prob original image
        if np.random.random() < self.p:
            return list_data[-1]
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

if __name__ == '__main__':
    im_path = r"D:\VNG\E2EObjectDetection\PolypsSet\train2019\Image"
    lb_path = r"D:\VNG\E2EObjectDetection\PolypsSet\train2019\Annotation"
    import xml.etree.ElementTree as ET
    from time import time
    NAME2IDX = {'hyperplastic':0, 'adenomatous':1}
    IDX2NAME = ['hyperplastic', 'adenomatous']
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

    def show(image, boxes, name='im'):
        for box in boxes:
            xmin, ymin, xmax, ymax = [int(p) for p in box[:4]]
            label = box[4]
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), (0, 255, 0), -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow(name, image)

    visual = VisualAugmenter(keep_prob=0.3)
    misc = MiscAugmenter(keep_prob=0.3)
    advanced = AdvancedAugmenter(keep_prob=0.5)
    im_names = os.listdir(im_path)
    for im_name in im_names:
        fp = os.path.join(im_path, im_name)
        xp = os.path.join(lb_path, im_name[:-3] + 'xml')
        if not os.path.exists(xp):
            continue
        boxes, names = parse_xml(xp)
        class_ids = [NAME2IDX[name] for name in names]
        img = cv2.imread(fp)
        ## samples 3 more data
        list_data = [{'image':img, 'boxes':boxes, 'class_ids':class_ids}]
        while len(list_data) != 4:
            
            temp_name = np.random.choice(im_names)
            fp = os.path.join(im_path, temp_name)
            xp = os.path.join(lb_path, temp_name[:-3] + 'xml')

            if not os.path.exists(xp):
                continue
            img = cv2.imread(fp)
            boxes, names = parse_xml(xp)
            list_data.append({'image':img, 'boxes':boxes, 'class_ids':class_ids})
        
        item_res = advanced(list_data)
        item_res = visual(item_res)
        # item_res = visual({'image':img, 'boxes':boxes, 'class_ids':class_ids})
        item_res = misc(item_res)
        # item_res = misc({'image':img, 'boxes':boxes, 'class_ids':class_ids})
        print(list_data[-1]['boxes'], list_data[-1]['class_ids'])
        print(item_res['boxes'], item_res['class_ids'])
        show(item_res['image'], [list(box) + [IDX2NAME[c]] for box, c in zip(item_res['boxes'], item_res['class_ids'])], 'transformed')
        show(img, [box + [c] for box, c in zip(boxes, names)], 'original')
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()


