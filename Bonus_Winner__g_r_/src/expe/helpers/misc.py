from PIL import Image
from itertools import chain
import numpy as np

def contrast(nd):
    m = nd.min()
    M = nd.max()
    return ((nd.astype(np.float32) - m) * 255 / max(1,(M-m)) ).astype(np.uint8)

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def display(nd):
    pilimg = Image.fromarray(nd)
    pilimg.show()

def makebox(keypoint):
    x,y = map(int,keypoint.pt)
    size = int(keypoint.size/2)
    #counter clockwise
    return [(x-size,y-size), (x-size,y+size), (x+size,y+size), (x+size,y-size)]