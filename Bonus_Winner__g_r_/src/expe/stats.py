# collect some stats
from itertools import islice

from helpers.input_data import Dataset


def get_bounding_box_stats(dataset):

    collect = []
    for scan_id, slice_idx, contour in dataset.get_contours():
        lft, top = contour.min(axis=(0,1))
        rgt, bot = contour.max(axis=(0,1))
        collect.append((scan_id, slice_idx, top, bot, lft, rgt))

    print("min top: %s"    % str(min(collect,key=lambda x:x[2])))
    print("max bot: %s"    % str(max(collect,key=lambda x:x[3])))
    print("min left: %s"   % str(min(collect,key=lambda x:x[4])))
    print("max rigth: %s"  % str(max(collect,key=lambda x:x[5])))


if __name__ == '__main__':

    dataset = Dataset("/home/gerey/hms_lung/data/example_extracted")
    get_bounding_box_stats(dataset)
