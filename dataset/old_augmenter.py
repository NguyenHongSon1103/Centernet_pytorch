import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from .transform import translation_xy, change_transform_origin, scaling_xy

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


def crop(image, boxes, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    h, w = image.shape[:2]
    min_x1, min_y1 = np.min(boxes, axis=0)[:2]
    max_x2, max_y2 = np.max(boxes, axis=0)[2:]
    random_x1 = np.random.randint(0, max(min_x1 // 2, 1))
    random_y1 = np.random.randint(0, max(min_y1 // 2, 1))
    random_x2 = np.random.randint(max_x2, max(min(w, max_x2 + (w - max_x2) // 2), max_x2 + 1))
    random_y2 = np.random.randint(max_y2, max(min(h, max_y2 + (h - max_y2) // 2), max_y2 + 1))
    image = image[random_y1:random_y2, random_x1:random_x2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - random_x1
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - random_y1
    return image, boxes


def flipx(image, boxes, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    image = image[:, ::-1]
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
    boxes = np.round(boxes * scale).astype(np.int32)
    return image, boxes


def translate(image, boxes, prob=0.5, border_value=(128, 128, 128)):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    h, w = image.shape[:2]
    min_x1, min_y1 = np.min(boxes, axis=0)[:2]
    max_x2, max_y2 = np.max(boxes, axis=0)[2:]
    translation_matrix = translation_xy(min=(min(-min_x1 // 2, 0), min(-min_y1 // 2, 0)),
                                        max=(max((w - max_x2) // 2, 1), max((h - max_y2) // 2, 1)), prob=1.)
    translation_matrix = change_transform_origin(translation_matrix, (w / 2, h / 2))
    image = cv2.warpAffine(
        image,
        translation_matrix[:2, :],
        dsize=(w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        points = translation_matrix.dot([
            [x1, x2, x1, x2],
            [y1, y2, y2, y1],
            [1, 1, 1, 1],
        ])
        min_x, min_y = np.min(points, axis=1)[:2]
        max_x, max_y = np.max(points, axis=1)[:2]
        new_boxes.append([min_x, min_y, max_x, max_y])
    boxes = np.array(new_boxes)
    return image, boxes


class MiscEffect:
    def __init__(self, multi_scale_prob=0.5, rotate_prob=0.05, flip_prob=0.5, crop_prob=0.5, translate_prob=0.5,
                 border_value=(0, 0, 0)):
        self.multi_scale_prob = multi_scale_prob
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.translate_prob = translate_prob
        self.border_value = border_value

    def __call__(self, image, boxes):
        image, boxes = multi_scale(image, boxes, prob=self.multi_scale_prob)
        image, boxes = rotate(image, boxes, prob=self.rotate_prob, border_value=self.border_value)
        image, boxes = flipx(image, boxes, prob=self.flip_prob)
        image, boxes = crop(image, boxes, prob=self.crop_prob)
        image, boxes = translate(image, boxes, prob=self.translate_prob, border_value=self.border_value)
        return image, boxes

def autocontrast(image, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.autocontrast(image)
    image = np.array(image)[..., ::-1]
    return image


def equalize(image, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.equalize(image)
    image = np.array(image)[..., ::-1]
    return image


def solarize(image, prob=0.5, threshold=128.):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.solarize(image, threshold=threshold)
    image = np.array(image)[..., ::-1]
    return image


def sharpness(image, prob=0.5, min=0, max=2, factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def color(image, prob=0.5, min=0., max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def contrast(image, prob=0.5, min=0.2, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def brightness(image, prob=0.5, min=0.8, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


class VisualEffect:
    """
    Struct holding parameters and applying image color transformation.

    Args
        solarize_threshold:
        color_factor: A factor for adjusting color.
        contrast_factor: A factor for adjusting contrast.
        brightness_factor: A factor for adjusting brightness.
        sharpness_factor: A factor for adjusting sharpness.
    """

    def __init__(
            self,
            color_factor=None,
            contrast_factor=None,
            brightness_factor=None,
            sharpness_factor=None,
            color_prob=0.5,
            contrast_prob=0.5,
            brightness_prob=0.5,
            sharpness_prob=0.5,
            autocontrast_prob=0.5,
            equalize_prob=0.5,
            solarize_prob=0.1,
            solarize_threshold=128.,

    ):
        self.color_factor = color_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.sharpness_factor = sharpness_factor
        self.color_prob = color_prob
        self.contrast_prob = contrast_prob
        self.brightness_prob = brightness_prob
        self.sharpness_prob = sharpness_prob
        self.autocontrast_prob = autocontrast_prob
        self.equalize_prob = equalize_prob
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold

    def __call__(self, image):
        """
        Apply a visual effect on the image.

        Args
            image: Image to adjust
        """
        random_enhance_id = np.random.randint(0, 4)
        if random_enhance_id == 0:
            image = color(image, prob=self.color_prob, factor=self.color_factor)
        elif random_enhance_id == 1:
            image = contrast(image, prob=self.contrast_prob, factor=self.contrast_factor)
        elif random_enhance_id == 2:
            image = brightness(image, prob=self.brightness_prob, factor=self.brightness_factor)
        else:
            image = sharpness(image, prob=self.sharpness_prob, factor=self.sharpness_factor)

        random_ops_id = np.random.randint(0, 3)
        if random_ops_id == 0:
            image = autocontrast(image, prob=self.autocontrast_prob)
        elif random_ops_id == 1:
            image = equalize(image, prob=self.equalize_prob)
        else:
            image = solarize(image, prob=self.solarize_prob, threshold=self.solarize_threshold)
        return image

class OldTransformer:
    def __init__(self):
        self.visual_augmenter = VisualEffect(color_prob=0.2, contrast_prob=0.2,
            brightness_prob=0.2, sharpness_prob=0.2, autocontrast_prob=0.2,
            equalize_prob=0.2, solarize_prob=0.2)
        self.misc_augmenter = MiscEffect(multi_scale_prob=0.2, rotate_prob=0.2,
                                         flip_prob=0.2, crop_prob=0.2,
                                         translate_prob=0.2)
    def __call__(self, item):
        '''
        item: a dictionary with {'image': image, 'boxes': boxes, 'class_ids':class_ids}
        '''
        
        item['image'] = self.visual_augmenter(item['image'])
        item['image'], item['boxes'] = self.misc_augmenter(item['image'], item['boxes'])
        return item
