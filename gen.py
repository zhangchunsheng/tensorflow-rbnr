#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw

NUMOBJS_DIR = "./numobjs"

def make_numobjs_ims(numobjs_path):
    im = numobjs_path
    yield numpy.array(im)[:, :, 0].astype(numpy.float32)

def generate_height():
    return random.randrange(30, 200);


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_COLOR)

    return bg


def generate_im(numobjs, num_bg_images):
    bg = generate_bg(num_bg_images)



def load_numobjs(folder_path):
    numobjs = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    return numobjs


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    numobjs = load_numobjs(NUMOBJS_DIR)
    num_bg_images = len(os.listdir("bgs"))
    while True:
        yield generate_im(numobjs, num_bg_images)


if __name__ == "__main__":
    os.mkdir("test")
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    print(enumerate(im_gen))
    for img_idx, (im) in enumerate(im_gen):
        fname = "test/{:08d}.png".format(img_idx)
        print(fname)
        cv2.imwrite(fname, im)

