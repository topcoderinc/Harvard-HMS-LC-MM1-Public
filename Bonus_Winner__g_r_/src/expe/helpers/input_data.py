# Copyright 2017 Yves Gerey. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

""" Build HM Lung Data """
import os

import numpy as np
import cv2

from collections import deque
from helpers.contours import read_coords

# TODO: pass them as parameters
ORIGINAL_SIZE = 512
TARGET_SIZE = 64


def grouper(iterable, n):
    "Collect data into fixed-length chunks, except for the last"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip(*args)


def batch(it, batch_size):
    for st in grouper(it, batch_size):
        yield stack(st)


def stack(samples_and_targets):
    data, targets = zip(*list(samples_and_targets))
    return np.stack(data), np.stack(targets)


class Scan():

    def __init__(self, dirpath, withGT=True):

        if withGT:
            with open(os.path.join(dirpath, "structures.dat")) as f:
                structures = f.read().strip().split('|')
                structure_ids = [i+1 for i,s in enumerate(structures)
                                 if "_gtv" in s.lower()]

        # NB: using dict avoids alphanumeric sorting issues
        slices = { int(os.path.splitext(f.name)[0]) : f.path
                   for f in os.scandir(os.path.join(dirpath,"pngs"))
                   if f.name.lower().endswith('.png') }
        contours = {}
        aux = {}
        # only keep path to tumor contours
        for id in slices:
            aux[id] = os.path.join(dirpath, 'auxiliary', str(id)+'.dat')
            if withGT:
                acc = []
                for f in os.scandir(os.path.join(dirpath, 'contours')):
                    slide_id, struct_id, ext = f.name.strip().split('.')
                    if int(slide_id) == id and \
                                    int(struct_id) in structure_ids and \
                                    ext.lower() == "dat":
                        acc.append(f.path)
                contours[id] = acc

        self.dirpath = dirpath
        self.slices = slices
        self.contours = contours
        self.aux = aux

        self.slice_index = 1
        self.queue = deque()
        self.CONTEXT_DEPTH = 3


    def id(self):
        return os.path.split(self.dirpath)[-1]


    def nb_slices(self):
        return len(self.slices)


    def _load_next_image(self):
        slicepath = self.slices[self.slice_index]
        self.slice_index +=1

        #png to raw array of uint16
        return cv2.imread(slicepath, cv2.IMREAD_UNCHANGED)


    def _gen_contours(self, slice_id):

        contourspaths = self.contours[slice_id]
        auxpath       = self.aux[slice_id]

        for path in contourspaths:
            # For each polygone (typicaly one)
            # TODO: directly pass list of polygones to fillPoly
            for coords in read_coords(path, auxpath):
                yield np.array(coords, np.int32).reshape((-1, 1, 2))


    def _build_target(self, slice_id, resize):
        # NB: fillPoly doesn't handle binary.
        # We'll resize and convert to float anyway.
        target = np.zeros((ORIGINAL_SIZE, ORIGINAL_SIZE), dtype=np.uint8)

        for points in self._gen_contours(slice_id):
            target = cv2.fillPoly(target, [points], 255)

        if resize:
            ratio = TARGET_SIZE/ORIGINAL_SIZE
            target = cv2.resize(target, (0, 0), fx=ratio, fy=ratio)
        return target.astype(np.float32) / 255


    def _has_tumor(self, slice_id):
        contourspaths = self.contours[slice_id]
        return 1 if contourspaths else 0


    def images_aux(self):
        """ return one single slice with aux data """

        while self.slice_index <= self.nb_slices():
           img = self._load_next_image()
           yield self.slice_index-1, img, self.aux


    def _sliced_input(self):

        if not self.queue:
            for _ in range(self.CONTEXT_DEPTH):
                self.queue.append(self._load_next_image())
        else:
            self.queue.popleft()
            self.queue.append(self._load_next_image())

        # We pack 3 consecutive slices as pixel features for
        # the middle ones.
        # TODO: use convolution approach instead
        #       to be more memory/speed efficient
        features = np.dstack(self.queue)

        middle = self.slice_index - 1 - (self.CONTEXT_DEPTH//2)

        return features, middle


    def features_and_targets(self, resize=True, filterBlank=False):
        """ return input (3 sliced) + target """

        while self.slice_index <= self.nb_slices():
            #TODO: optimize (don't load image if case of filtering)
            features, slice_id = self._sliced_input()
            target = self._build_target(slice_id, resize)
            if not filterBlank or target.sum():
                yield self.id(), slice_id, features, target


    def features_and_binary_targets(self):
        """ return input (3 sliced), 1 if scans contains tumor else 0 """

        while self.slice_index <= self.nb_slices():
             features, slice_id = self._sliced_input()
             target = self._has_tumor(slice_id)
             yield self.id(), slice_id, features, target


    def images_and_targets(self, resize=False):

        while self.slice_index <= self.nb_slices():
            img = self._load_next_image()
            target = self._build_target(self.slice_index-1, resize)
            yield self.id(), self.slice_index-1, self.nb_slices(), img, target


    def get_contours(self):  # contours already used as attribute!

        while self.slice_index <= self.nb_slices():
            for contour in self._gen_contours(self.slice_index):
                yield self.id(), self.slice_index, contour
            self.slice_index += 1


    def images_and_aux(self):

        while self.slice_index <= self.nb_slices():
            img = self._load_next_image()
            slice_id = self.slice_index-1
            yield self.id(), slice_id, self.nb_slices(), img, self.aux[self.slice_id]

    def gen_aux(self):
        while self.slice_index <= self.nb_slices():
            yield self.id(), self.slice_index, self.aux[self.slice_index]
            self.slice_index += 1


    def scan3D(self):

        res = np.stack(self._load_next_image() for _ in range(self.nb_slices()) )
        #rembobine
        self.slice_index = 1
        return res



class Dataset():

    def __init__(self, dirpath, withGT=True):
        scandirs = [ d.path for d in os.scandir(dirpath)
                     if not d.name.startswith('.') and d.is_dir()]

        # collect all paths of interest
        self.withGT = withGT
        self.scandirs = scandirs

    def nb_scans(self):
        return len(self.scandirs)

    def scans(self):

        for scandir in self.scandirs:
            yield Scan(scandir, self.withGT)


    def __getattr__(self, methodname):
        """ map method name to all scans"""

        def method(**kwargs):
            for scan in self.scans():
                yield from getattr(scan, methodname)(**kwargs)
        return method






