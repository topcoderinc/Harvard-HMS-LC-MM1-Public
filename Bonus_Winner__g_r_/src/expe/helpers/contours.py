# Copyright 2017 Yves Gerey. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

""" Read coords from CSV and convert them in pixel """

from itertools import zip_longest
import numpy as np
import cv2

IMAGE_SIZE = 512  # TODO, DRY & more flexibility

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def cv2tolist(contour):
    """ Take a contour as returned by opencv2 into a list of (x,y) """
    return contour.reshape((-1,2)).tolist()

def merge_contours_naive(contours):
    win = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)  # bool not supported by cv2

    # For each polygone (typical one)
    # TODO: directly pass list of polygones to fillPoly?
    for cnt in contours:
        points = np.array(cnt, np.int32).reshape((-1, 1, 2))
        win = cv2.fillPoly(win, [points], 1)

    img2, contours, hierarchy = cv2.findContours(win ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #TODO: ensure counter-clockwise
    return [cv2tolist(cnt) for cnt in contours]

def getdat(datpath, memo={}):

    if datpath in memo:
        return memo[datpath]

    dat = {}
    with open(datpath, "r") as f:
        for l in f.readlines():
            k, *v = l.strip().split(',')
            dat [k] = v

    memo[datpath] = dat
    return dat


def read_coords(csvpath, datpath):

    with open(csvpath, "r") as f:
        values = [ [float(x) for x in l.strip().split(',')] for l in f.readlines() ]

    # drop the z (constant for each slice)
    return [ convert_coords( ((x,y) for x,y,z in list(grouper(v, 3))),
                             datpath) for v in values ]


def convert_coords(contour, datpath):

    dat = getdat(datpath)
    x0, y0, z = map(float, dat['(0020.0032)'])
    dx, dy    = map(float, dat['(0028.0030)'])

    def mmToPixel(x,y):
        return (x-x0)/dx, (y-y0)/dy

    return [ mmToPixel(x,y) for x, y in contour ]


def pixelToMM(datpath, x,y):

    dat = getdat(datpath)
    x0, y0, z = map(float, dat['(0020.0032)'])
    dx, dy    = map(float, dat['(0028.0030)'])
    return x*dx + x0, y*dy + y0


def draw_contours(img, cnts):

    for cnt in cnts:
        points = np.array(cnt, np.int32).reshape((-1, 1, 2))
        cv2.drawContours(img, [points], -1, (255,155,0), 1)
