import numpy as np
import os, sys
import tensorflow as tf
import cv2

MODEL_ROOT = "/Users/changetheworld/dev/git/tensorflow-rbnr"
sys.path.append(MODEL_ROOT)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = MODEL_ROOT + '/models/frozen_inference_graph.pb'  # frozen model path
PATH_TO_LABELS = os.path.join(MODEL_ROOT, 'labels', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

'''PATH_TO_CKPT = MODEL_ROOT + '/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'  # frozen model path
PATH_TO_LABELS = os.path.join(MODEL_ROOT, 'labels', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

PATH_TO_CKPT = MODEL_ROOT + '/faster_rcnn_resnet50_coco_2017_11_08/frozen_inference_graph.pb'  # frozen model path
PATH_TO_LABELS = os.path.join(MODEL_ROOT, 'labels', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

PATH_TO_CKPT = MODEL_ROOT + '/faster_rcnn_resnet101_kitti_2017_11_08/frozen_inference_graph.pb'  # frozen model path
PATH_TO_LABELS = os.path.join(MODEL_ROOT, 'labels', 'kitti_label_map.pbtxt')
NUM_CLASSES = 2

PATH_TO_CKPT = MODEL_ROOT + '/faster_rcnn_inception_resnet_v2_atrous_oid_2017_11_08/frozen_inference_graph.pb'  # frozen model path
PATH_TO_LABELS = os.path.join(MODEL_ROOT, 'labels', 'oid_bbox_trainable_label_map.pbtxt')
NUM_CLASSES = 546'''

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def detect(image_path):
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
          image = cv2.imread(image_path)
          image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=4)
          new_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
          cv2.imshow("test", new_img)
          cv2.waitKey(0)

if __name__ == '__main__':
    detect("./sports/1509882717793_d80e5584_4ac7_4e77_ad95_dddf7e64e572.jpg")