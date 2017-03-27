from collections import defaultdict
import os
import skimage
import skimage.draw
import numpy as np
import skimage.io
import json
from skimage import measure
import math


CLASS_WEIGHTS = [1.0, 3.0, 6.0, 9.0, 12.0, 15.0,
                 18.0, 8.0, 24.0]

LUNG_CLASS_WEIGHTS = [1.0, 2.0]

STRUCTURE_CLASS = [('body', ('body')),
                   ('heart',('heart')),
                   ('lung', ('lung')),
                   ('bronchi', ('bronchi')),
                   ('esophagus', ('esophagus')),
                   ('trachea', ('trachea')),
                   ('radiomics_gtv', ('radiomics_gtv')),
                   ('radiomics_ln', ('radiomics_ln'))]

LUNG_CLASSES = [('lungs', ('lung', 'radiomics_gtv'))]

INPUT_SHAPE = (512, 512)
INPUT_LUNG_SHAPE = (256, 256)


class Prediction:
    def __init__(self, scan_id, slice_id, predictions, probabilities):
        self.scan_id = scan_id
        self.slice_id = slice_id
        self.predictions = predictions
        self.probabilities = probabilities


def mm_to_pixels(x, y, x0, y0, dx, dy):
    rx = (x - x0) / dx
    ry = (y - y0) / dy
    return rx, ry


def pixels_to_mm(x, y, x0, y0, dx, dy):
    rx = x * dx + x0
    ry = y * dy + y0
    return rx, ry


def parse_line(line):
    return json.loads(line)


def read_annotations(filename):
    result = defaultdict(lambda: {})
    with open(filename, 'r') as f:
        for line in f:
            slice_data = parse_line(line)
            result[slice_data['scan_id']][slice_data['slice_id']] = slice_data
    return result


def get_image_filename(root_dir, annotation_data):
    path = os.path.join(root_dir, annotation_data['scan_id'],
                        'pngs', annotation_data['slice_id'] + '.png')
    return path


def read_image(image_filename):
    image = skimage.io.imread(image_filename)
    image = image / 65536
    return image


def draw_contours(image_data, contours):
    image_data = np.copy(image_data)
    for contour in contours:
        xs = [x for i, x in enumerate(contour) if i % 2 == 0]
        ys = [x for i, x in enumerate(contour) if i % 2 == 1]
        rr, cc = skimage.draw.polygon_perimeter(ys, xs)
        image_data[rr, cc] = 0.0
    return image_data


def create_mask(image_shape, structures):
    mask_image = np.zeros(image_shape)
    for i, (structure_name, v) in enumerate(STRUCTURE_CLASS):
        cls_index = i + 1
        if structure_name in structures:
            contours = structures[structure_name]
            mask_image = mask_contours(mask_image, contours, cls_index)
    return mask_image


def create_joint_mask(image_shape, structures, class_descriptions):
    mask_image = np.zeros(image_shape)
    for i, (k, structure_names) in enumerate(class_descriptions):
        cls_index = i + 1
        for structure_name in structure_names:
            if structure_name in structures:
                contours = structures[structure_name]
                mask_image = mask_contours(mask_image, contours, cls_index)
    return mask_image


def mask_contours(image_data, contours, index):
    mask_image = np.zeros_like(image_data)
    for contour in contours:
        xs = [x for i, x in enumerate(contour) if i % 2 == 0]
        ys = [x for i, x in enumerate(contour) if i % 2 == 1]
        rr, cc = skimage.draw.polygon(ys, xs)
        mask_image[rr, cc] = index
    return mask_image


def mask_single_contour(image_data, xs, ys, index):
    mask_image = np.zeros_like(image_data)
    rr, cc = skimage.draw.polygon(ys, xs)
    mask_image[rr, cc] = index
    return mask_image


def get_contour_points(mask_image):
    contour_points = measure.find_contours(mask_image, 0.5)
    result = [(list(c[:, 1]), list(c[:, 0])) for c in contour_points]
    return result


def sparse_tensor_to_strings(tensor, batch_size):
    s = []
    for i in range(batch_size):
        labels = tensor.values[tensor.indices[:, 0] == i]
        s.append(labels[0].decode())
    return s


def split_batch(scan_ids, slice_ids, predictions, probabilities):
    batch_size = predictions.shape[0]
    predictions = np.split(predictions, predictions.shape[0])
    probabilities = np.split(probabilities, probabilities.shape[0])
    scan_ids = sparse_tensor_to_strings(scan_ids, batch_size)
    slice_ids = sparse_tensor_to_strings(slice_ids, batch_size)
    result = []
    for scan_id, slice_id, preds, probs in zip(scan_ids, slice_ids, predictions, probabilities):
        result.append(Prediction(scan_id, slice_id, preds[0], probs[0]))
    return result


def polygon_area(xs, ys):
    area = 0.0
    for i in range(len(xs)):
        area += xs[i] * ys[(i + 1) % len(xs)] - ys[i] * xs[(i + 1) % len(xs)]
    return math.fabs(area) / 2


def draw_prediction_contour(image):
    contour = measure.find_contours(image, 0.2)
    contour_image = np.zeros_like(image)
    contours = []
    for i in range(len(contour)):
        rr, cc = list(contour[i][:, 0]), list(contour[i][:, 1])
        area = polygon_area(cc, rr)
        contours.append((area, rr, cc))
    contours = sorted(contours, key=lambda x: -x[0])
    area, rr, cc = contours[0]
    print(area)
    contour_image[rr, cc] = 1.0

    return contour_image


def get_prediction_contour(image):
    contour = measure.find_contours(image, 0.2)
    contours = []
    for i in range(len(contour)):
        rr, cc = list(contour[i][:, 0]), list(contour[i][:, 1])
        area = polygon_area(cc, rr)
        contours.append((area, rr, cc))
    contours = sorted(contours, key=lambda x: -x[0])
    if not contours:
        return None
    result = [(rr, cc) for area, rr, cc in contours]
    return result


def get_image_range(input_dir, scan_id, start_slice_id, end_slice_id, zeros_shape):
    images = []
    for slice_id in range(start_slice_id, end_slice_id + 1):
        image_path = os.path.join(input_dir, str(scan_id), 'pngs', str(slice_id) + '.png')
        if not os.path.exists(image_path):
            images.append(np.zeros(zeros_shape))
        else:
            images.append(read_image(image_path))
    return images


def get_expanded_bounding_box(p1, p2, target_shape):
    height = p2[0] - p1[0]
    h1 = (target_shape[0] - height + 1) // 2
    h2 = (target_shape[0] - height) // 2
    start_r = p1[0] - h1
    end_r =  p2[0] + h2

    width = p2[1] - p1[1]
    w1 = (target_shape[1]- width + 1) // 2
    w2 = (target_shape[1] - width) // 2
    start_c = p1[1] - w1
    end_c =  p2[1] + w2

    return (start_r, start_c), (end_r, end_c)

def get_bounding_box(mask, threshold_small_regions=50):
    regions = measure.find_contours(mask, 0.5)
    min_r, max_r, min_c, max_c = [mask.shape[0] / 2] * 4
    for reg in regions:
        rr, cc = reg[:, 0], reg[:, 1]
        area = polygon_area(rr, cc)
        if area >= threshold_small_regions:
            min_r = int(min(min_r, np.min(rr)))
            max_r = int(max(max_r, np.max(rr)))
            min_c = int(min(min_c, np.min(cc)))
            max_c = int(max(max_c, np.max(cc)))
    return (min_r, min_c), (max_r, max_c)

