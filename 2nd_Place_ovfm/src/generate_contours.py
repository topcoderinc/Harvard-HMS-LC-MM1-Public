import tensorflow as tf
import argparse
import os
import utils
from multiprocessing import Pool, cpu_count
import json
import pickle
import numpy as np
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert to dataset.')
    parser.add_argument(
        '--predictions_dir', type=str, required=True,
        help='Directoy with predictions.')
    parser.add_argument(
        '--input_annotations', type=str, required=True,
        help='Input annotations file.')
    parser.add_argument(
        '--input_lungs_dir', type=str, required=True,
        help='Input directory of lungs.')
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Input directory of lungs.')
    parser.add_argument(
        '--output_contours_path', type=str, required=True,
        help='Output contours file.')
    args = parser.parse_args()

    annotations = defaultdict(dict)
    write_file = open(args.output_contours_path, 'w')
    with open(args.input_annotations, 'r') as f:
        for line in f:
            data = json.loads(line)
            annotations[data['scan_id']][data['slice_id']] = data

    for scan_id, slice_annotations in annotations.items():
        for slice_id, data in slice_annotations.items():
            predictions_file = os.path.join(args.predictions_dir, data['scan_id'] + '.' + data['slice_id'])
            with open(predictions_file, 'rb') as pf:
                predictions = pickle.load(pf)
            with open(os.path.join(args.input_lungs_dir, data['scan_id'], 'coordinates.txt'), 'r') as f:
                r0, c0, r1, c1 = [int(x) for x in f.read().split(",")]

            image_file_path = os.path.join(args.input_dir, data['scan_id'], 'pngs', data['slice_id'] + '.png')

            input_image = utils.read_image(image_file_path)

            input_image = input_image / 0.0625

            foreground = input_image > 0.15

            # predictions = probabilities > 0.5
            segmentation_predictions = predictions[0]

            bigger_mask = np.zeros((512, 512), np.uint8)
            bigger_mask[r0:r1, c0:c1] = segmentation_predictions
            bigger_mask = bigger_mask * foreground

            bigger_probs = np.zeros((512, 512), np.float)
            bigger_probs[r0:r1, c0:c1] = predictions[1]

            segmentation_predictions = bigger_mask

            radiomics_preds = segmentation_predictions
            radiomics_preds = (
                segmentation_predictions ==
                (utils.STRUCTURE_CLASS.index(('radiomics_gtv', ('radiomics_gtv'))) + 1)).astype(np.int32)
            contours = utils.get_prediction_contour(radiomics_preds)

            if not contours:
                continue

            region_probabilities = []
            filtered_contours = []
            for contour in contours:
                ys, xs = contour
                region = utils.mask_single_contour(bigger_mask, xs, ys, 1)
                area = np.sum(region)
                average_prob = np.sum(bigger_probs * region) / area
                region_probabilities.append((average_prob, area))
                if average_prob > 0.0 and area > 20:
                    filtered_contours.append(contour)

            print("Region prob:",  region_probabilities)


            print("COntours", len(contours), len(filtered_contours))
            for contour in filtered_contours:
                ys, xs = contour
                mm_coordinates = []
                for x, y in zip(xs, ys):
                    x, y = utils.pixels_to_mm(x, y, data['tags']['x0'], data['tags']['y0'],
                                              data['tags']['dx'], data['tags']['dy'])
                    mm_coordinates.append(x)
                    mm_coordinates.append(y)

                write_file.write('{},{},{}'.format(data['scan_id'], data['slice_id'], ",".join([str(x) for x in mm_coordinates])))

                write_file.write('\n')

    write_file.close()
