import numpy as np
import albumentations as A
import cv2

ROTATE_DEGREE = [90, 180, 270]

def rotate(image, boxes, prob=0.5, border_value=(128, 128, 128)):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    rotate_degree = ROTATE_DEGREE[np.random.randint(0, 3)]
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=border_value)
    if boxes is None:
        return image, None

    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        points = M.dot([
            [x1, x2, x1, x2],
            [y1, y2, y2, y1],
            [1, 1, 1, 1],
        ])

        # Extract the min and max corners again.
        min_xy = np.sort(points, axis=1)[:, :2]
        min_x = np.mean(min_xy[0])
        min_y = np.mean(min_xy[1])
        max_xy = np.sort(points, axis=1)[:, 2:]
        max_x = np.mean(max_xy[0])
        max_y = np.mean(max_xy[1])

        new_boxes.append([min_x, min_y, max_x, max_y])
    boxes = np.array(new_boxes)
    return image, boxes

def flipx(image, boxes, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    image = image[:, ::-1]
    if boxes is None:
        return image, None
    h, w = image.shape[:2]
    tmp = boxes[:, 0].copy()
    boxes[:, 0] = w - boxes[:, 2]
    boxes[:, 2] = w - tmp
    return image, boxes

def multi_scale(image, boxes, prob=1.):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    h, w = image.shape[:2]
    scale = np.random.choice(np.arange(0.7, 1.4, 0.1))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if boxes is None:
        return image, None
    boxes = np.round(boxes * scale).astype(np.int32)
    return image, boxes

class WeakAug:
    def __init__(self, multi_scale_prob=0.5, rotate_prob=0.05, flip_prob=0.5,
                 border_value=(0, 0, 0)):
        self.multi_scale_prob = multi_scale_prob
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.border_value = border_value

    def __call__(self, image, boxes=None):

        image, boxes = multi_scale(image, boxes, prob=self.multi_scale_prob)
        image, boxes = rotate(image, boxes, prob=self.rotate_prob, border_value=self.border_value)
        image, boxes = flipx(image, boxes, prob=self.flip_prob)
        if boxes is None:
            return image
        return image, boxes

class StrongAug:
    def __init__(self):
        '''
        Augmentation methods used: ColorJitter, Grayscale, GaussianBlur, RandomCrop
        '''
        self.transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.CoarseDropout (max_holes=1, max_height=50, max_width=50,
                                        min_height=10, min_width=10, p=0.7),
            A.CoarseDropout (max_holes=1, max_height=100, max_width=100,
                                        min_height=30, min_width=30, p=0.5),
            A.CoarseDropout (max_holes=1, max_height=200, max_width=200,
                                        min_height=50, min_width=50, p=0.3),
            A.GaussianBlur(sigma_limit=2.0, p=0.5),
            A.ToGray(p=0.2)
        ])

    def __call__(self, image):
        image = self.transform(image=image)['image']
        if len(image.shape) == 2:
            image = np.stack([image, image, image], -1)
        return image