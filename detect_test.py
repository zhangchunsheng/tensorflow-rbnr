#!/usr/bin/env python

import cv2;
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = "/Users/changetheworld/dev/git/tensorflow-rbnr/models/frozen_inference_graph.pb";
        self.PATH_TO_LABELS = "/Users/changetheworld/dev/git/tensorflow-rbnr/labels/mscoco_label_map.pbtxt";
        self.NUM_CLASSES = 1
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph);
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_np_extended = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_extended})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8
                )

        cv2.namedWindow('detection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('detection', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    image = cv2.imread('./sports/1509882717793_d80e5584_4ac7_4e77_ad95_dddf7e64e572.jpg')
    detector = TOD()
    detector.detect(image)