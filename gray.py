#!/usr/bin/env python

import os
import sys
import tarfile

import cv2
import numpy

if __name__ == "__main__":
    im = cv2.imread('./id__1_1000.JPG', cv2.IMREAD_GRAYSCALE);
    fname = "./id__1_1000_gray.JPG";
    cv2.imwrite(fname, im)

    im = cv2.imread('./id__2_1000.JPG', cv2.IMREAD_GRAYSCALE);
    fname = "./id__2_1000_gray.JPG";
    cv2.imwrite(fname, im)
