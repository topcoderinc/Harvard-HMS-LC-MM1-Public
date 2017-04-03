import argparse
import glob
import os
import pickle
import utils
from skimage.io import imsave


def process_scan(scan_path, predictions_dir, output_directory):
    p1, p2 = (256, 256), (256, 256)
    scan_id = os.path.basename(scan_path)
    for slice_image_path in glob.glob(scan_path + '/pngs/*.png'):
        slice_id = os.path.basename(slice_image_path).split(".")[0]

        with open(
                os.path.join(predictions_dir, '{}.{}'.format(scan_id, slice_id)),
                'rb') as f:
            mask = pickle.load(f)
            cp1, cp2 = utils.get_bounding_box(mask)
            p1 = min(p1[0], cp1[0]), min(p1[1], cp1[1])
            p2 = max(p2[0], cp2[0]), max(p2[1], cp2[1])
    p1, p2 = utils.get_expanded_bounding_box(p1, p2, (256, 256))

    scan_output_dir = os.path.join(output_directory, scan_id)
    os.makedirs(scan_output_dir + '/pngs')

    with open(os.path.join(output_directory, scan_id, "coordinates.txt"), 'w') as f:
        f.write("{},{},{},{}".format(p1[0], p1[1], p2[0], p2[1]))


    for slice_image_path in glob.glob(scan_path + '/pngs/*.png'):
        image = utils.read_image(slice_image_path)
        slice_id = os.path.basename(slice_image_path).split(".")[0]
        with open(
                os.path.join(predictions_dir, '{}.{}'.format(scan_id, slice_id)),
                'rb') as f:
            mask = pickle.load(f)
            new_image = (image * mask)[p1[0]:p2[0], p1[1]:p2[1]]
            output_image_path = os.path.join(output_directory, scan_id, 'pngs',
                                             os.path.basename(slice_image_path))
            imsave(output_image_path, new_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert to dataset.')
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Input directory.')
    parser.add_argument(
        '--output_directory', type=str, required=True,
        help='Output directory.')
    parser.add_argument(
        '--predictions_dir', type=str, required=False, default=None,
        help='Predictions directory.')
    args = parser.parse_args()

    for scan_path in glob.glob(args.input_dir + '/*'):
        process_scan(scan_path, args.predictions_dir, args.output_directory)
