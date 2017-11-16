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

import tensorflow as tf

from utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', './test/rbnr.record', 'Path to output TFRecord')
flags.DEFINE_string('draw_bbox', 0, 'Draw bbox')
flags.DEFINE_string('numbers', 128, 'Make numbers of images')
FLAGS = flags.FLAGS

DATA_DIR = "./test"
NUMOBJS_DIR = "./numobjs"
BGS_DIR = "./bgs"

def make_numobjs_ims(numobjs_path):
    numobj = cv2.imread(numobjs_path, cv2.IMREAD_COLOR)
    return numobj

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

def make_affine_transform(from_shape, to_shape,
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h = from_shape[0]
    w = from_shape[1]
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds

def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = BGS_DIR + "/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_COLOR)
        found = True

    return bg

def generate_im(numobj_im, num_bg_images):
    bg = generate_bg(num_bg_images)

    M, out_of_bounds = make_affine_transform(
        from_shape=numobj_im.shape,
        to_shape=bg.shape,
        min_scale=0.6,
        max_scale=0.875,
        rotation_variation=1.0,
        scale_variation=1.5,
        translation_variation=1.2)

    numobj_im1 = cv2.warpAffine(numobj_im, M, (bg.shape[1], bg.shape[0]))
    out = numobj_im1 + bg

    return out;

def generate_int(min, max):
    return random.randint(min, max);

def generate_float(min, max, step=1):
    return random.randrange(min, max, step) / 10.;

def scaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, expand = False):
    if center is None:
        return image.rotate(angle)
    angle = -angle / 180.0 * math.pi
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)

def draw_box(bg, x, y, width, height):
    draw = ImageDraw.Draw(bg)
    draw.rectangle([(x, y), (x + width, y + height)], None, (255, 255, 0))
    del draw

def draw_numobj(bg, obj):
    angle = generate_int(-10, 10);
    scalex = generate_float(5, 10);
    scaley = generate_float(5, 10);

    width = obj.size[0]
    height = obj.size[1]
    bg_width = bg.size[0]
    bg_height = bg.size[1]

    if(bg_width < width + 100):
        scalex = 0.3
    if(bg_height < height + 100):
        scaley = 0.3

    scale = (scalex, scaley);

    im = obj.convert('RGBA')
    # out = im.rotate(45, expand=1)
    # out = im.transpose(Image.ROTATE_180)
    out = scaleRotateTranslate(im, angle, (width / 2, height / 2), None, scale);

    box = out.getbbox();# left, upper, right, and lower
    x = generate_int(100, 200)
    y = generate_int(100, 200)
    w = box[2] - box[0]
    h = box[3] - box[1]
    angle = -angle / 180.0 * math.pi
    asin = math.asin(angle)
    rx = x + (width - width * scalex) / 2 + h * asin / 2;
    ry = y + (height - height * scaley) / 2 + w * asin / 2;
    if((rx + w) > bg_width):
        x = x - w;
        rx = x + (width - width * scalex) / 2 + h * asin / 2;
    if ((ry + h) > bg_height):
        y = y - h;
        ry = y + (height - height * scaley) / 2 + w * asin / 2;
    if(rx < 0):
        x = x + w;
        rx = x + (width - width * scalex) / 2 + h * asin / 2;
    if(ry < 0):
        y = y + h;
        ry = y + (height - height * scaley) / 2 + w * asin / 2;

    if(FLAGS.draw_bbox):
        draw_box(bg, rx, ry, w, h);

    bg.paste(out, (x, y), out)

    return [(bg_width, bg_height, rx, ry, w, h)]

def generate_impl(numobj_im, num_bg_images):
    bg = generate_bg(num_bg_images)

    pil_bg = Image.fromarray(bg, 'RGB')
    numobj_im_bg = Image.fromarray(numobj_im, 'RGB')

    bbox = draw_numobj(pil_bg, numobj_im_bg);

    return numpy.array(pil_bg), bbox;

def load_numobjs(folder_path):
    numobjs_ims = {}
    numobjs = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    for numobj in numobjs:
        numobjs_ims[numobj] = make_numobjs_ims(os.path.join(folder_path, numobj))
    return numobjs, numobjs_ims

def create_tf_example(example):
    bw, bh, rx, ry, w, h = example['bbox'][0]

    width = long(bw)  # Image width
    height = long(bh) # Image height
    filename = example['filename'] # Filename of the image. Empty if image is not from file

    with tf.gfile.GFile(example['fname'], 'rb') as fid:
        encoded_image_data = fid.read() # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'

    xmins = [rx / width] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [(rx + w) / width] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [ry / height] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [(ry + h) / height] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = ['yichang_rbn'] # List of string class name of bounding box (1 per box)
    classes = [1] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    numobjs, numobjs_ims = load_numobjs(NUMOBJS_DIR)
    num_bg_images = len(os.listdir(BGS_DIR))
    while True:
        yield generate_impl(numobjs_ims[random.choice(numobjs)], num_bg_images)

def showImg(im):
    cv2.namedWindow("image", 1)
    cv2.imshow("image", im)
    cv2.waitKey()
    cv2.destroyWindow("image")

def nukedir(dir):
    if dir[-1] == os.sep: dir = dir[:-1]
    files = os.listdir(dir)
    for file in files:
        if file == '.' or file == '..': continue
        path = dir + os.sep + file
        if os.path.isdir(path):
            nukedir(path)
        else:
            os.unlink(path)
    os.rmdir(dir)

if __name__ == "__main__":
    if os.path.exists(DATA_DIR):
        nukedir(DATA_DIR)
    os.mkdir(DATA_DIR)
    im_gen = itertools.islice(generate_ims(), int(FLAGS.numbers))
    print(im_gen);

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for img_idx, (im, bbox) in enumerate(im_gen):
        filename = "{:08d}.png".format(img_idx)
        fname = DATA_DIR + "/" + filename
        print(fname)
        cv2.imwrite(fname, im)
        #showImg(im)

        example = {
            "filename": filename,
            "fname": fname,
            "bbox": bbox
        }
        print(example)
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()

