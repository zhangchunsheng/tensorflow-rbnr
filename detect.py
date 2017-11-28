r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files may be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being evaluated, an
input_reader_pb2.InputReader file to specify what data the model is evaluating
and an eval_pb2.EvalConfig file to configure evaluation parameters.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=eval_config.pbtxt \
        --model_config_path=model_config.pbtxt \
        --input_config_path=eval_input_config.pbtxt

    python object_detection/eval.py \
        --logtostderr \
        --checkpoint_dir=/Users/changetheworld/dev/git/tensorflow-rbnr/models/modelNumobj/train \
        --eval_dir=/Users/changetheworld/dev/git/tensorflow-rbnr/models/modelNumobj/eval \
        --pipeline_config_path=/Users/changetheworld/dev/git/tensorflow-rbnr/models/modelNumobj/faster_rcnn_numobj.config
"""

import cv2;
import functools
import os
import tensorflow as tf

import logging
import numpy as np
import PIL.Image as Image

import evaluator
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.core import standard_fields as fields
from object_detection.utils import visualization_utils as vis_utils
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import keypoint_ops
from object_detection.utils import ops

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string('checkpoint_dir', '/Users/changetheworld/dev/git/tensorflow-rbnr/models/modelNumobj/train',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', '/Users/changetheworld/dev/git/tensorflow-rbnr/models/modelNumobj/eval',
                    'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', '/Users/changetheworld/dev/git/tensorflow-rbnr/models/modelNumobj/faster_rcnn_numobj.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('eval_config_path', '',
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
flags.DEFINE_boolean('run_once', True, 'Option to only run a single pass of '
                     'evaluation. Overrides the `max_evals` parameter in the '
                     'provided config.')
FLAGS = flags.FLAGS

def main(unused_argv):
    assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
    assert FLAGS.eval_dir, '`eval_dir` is missing.'
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    if FLAGS.pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(
            FLAGS.pipeline_config_path
        )
        tf.gfile.Copy(FLAGS.pipeline_config_path,
                      os.path.join(FLAGS.eval_dir, 'pipeline.config'),
                      overwrite=True)
    else:
        configs = config_util.get_configs_from_multiple_files(
            model_config_path=FLAGS.model_config_path,
            eval_config_path=FLAGS.eval_config_path,
            eval_input_config_path=FLAGS.input_config_path
        )
        for name, config in [('model.config', FLAGS.model_config_path),
                             ('eval.config', FLAGS.eval_config_path),
                             ('input.config', FLAGS.input_config_path)]:
            tf.gfile.Copy(config,
                          os.path.join(FLAGS.eval_dir, name),
                          overwrite=True)

    model_config = configs['model']
    eval_config = configs['eval_config']
    input_config = configs['eval_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False
    )

    label_map = label_map_util.load_labelmap(input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes)

    if FLAGS.run_once:
        eval_config.max_evals = 1

    detect(model_fn)

def detect(create_model_fn):
    model = create_model_fn()

    image = cv2.imread('./sports/1509882717793_d80e5584_4ac7_4e77_ad95_dddf7e64e572.jpg')
    image = cv2.imread('./eval/00000000.png', cv2.IMREAD_GRAYSCALE)
    original_image = tf.expand_dims(image, axis=0)
    preprocessed_image = model.preprocess(tf.to_float(original_image))
    prediction_dict = model.predict(preprocessed_image)
    detections = model.postprocess(prediction_dict)

    sess = tf.Session('', graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer());
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    if restore_fn:
        restore_fn(sess)
    else:
        if not FLAGS.checkpoint_dir:
            raise ValueError('`checkpoint_dirs` must have at least one entry')
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        variables_to_restore = tf.global_variables()
        global_step = tf.train.get_or_create_global_step()
        variables_to_restore.append(global_step)

        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, checkpoint_file)

    tensor_dict = result_dict_for_single_example(
        original_image,
        detections,
        class_agnostic=(
            fields.DetectionResultFields.detection_classes not in detections
        ),
        scale_to_absolute=True
    )

    result_dict = sess.run(tensor_dict)

    label_map = label_map_util.load_labelmap("/Users/changetheworld/dev/git/tensorflow-rbnr/labels/rbnr_label_map.pbtxt")
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes)

    visualize_detection_results(
        result_dict,
        categories=categories,
    )

def restore_fn(sess):
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)

    saver = tf.train.Saver(variables_to_restore)

    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    saver.restore(sess, latest_checkpoint)

def result_dict_for_single_example(image,
                                   detections,
                                   class_agnostic=False,
                                   scale_to_absolute=False):
    """Merges all detection and groundtruth information for a single example.

    Note the evaluation tools require classes that are 1-indexed, and so this
    function performs the offset. If `class_agnostic` is True, all output classes
    have label 1.

    Args:
        image: A single 4D image tensor of shape[1, H, W, C].
        key: A single string tensor identifying the image.
        detections: A dictionary of detections, returned from
            DetectionModel.postprocess().
        groundtruth: (Optional) Dictionary of groundtruth items, with fields:
            'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
                normalized coordinates.
            'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
            'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)
            'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)
            'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)
            'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)
            'groundtruth_instance_masks': 3D int64 tensor of instance masks
                (Optional).
        class_agnostic: Boolean indicating whether the detections are class-agnostic
            (i.e. binary). Default False.
        scale_to_absolute: Boolean indicating whether boxes, masks, keypoints should
            be scaled to absolute coordinates. Note that for IoU based evaluations,
            it does not matter whether boxes are expressed in absolute or relative
            coordinates. Default False.

    Returns:
        A dictionary with:
        'original_image': A [1, H, W, C] uint8 image tensor.
        'key': A string tensor with image identifier.
        'detection_boxes': [max_detections, 4] float32 tensor of boxes, in
            normalized or absolute coordinates, depending on the value of
            `scale_to_absolute`.
        'detection_scores': [max_detections] float32 tensor of scores.
        'detection_classes': [max_detections] int64 tensor of 1-indexed classes.
        'detection_masks': [max_detections, None, None] float32 tensor of binarized
            masks. (Only present if available in `detections`)
        'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
            normalized or absolute coordinates, depending on the value of
            `scale_to_absolute`. (Optional)
        'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
            (Optional)
        'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)
        'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)
        'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)
        'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)
        'groundtruth_instance_masks': 3D int64 tensor of instance masks
            (Optional).
    """
    label_id_offset = 1 # Applying label id offset (b/63711816)
    input_data_fields = fields.InputDataFields()
    output_dict = {
        input_data_fields.original_image: image,
    }

    detection_fields = fields.DetectionResultFields
    detection_boxes = detections[detection_fields.detection_boxes][0]
    output_dict[detection_fields.detection_boxes] = detection_boxes
    image_shape = tf.shape(image)
    if scale_to_absolute:
        absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
            box_list.BoxList(detection_boxes), image_shape[1], image_shape[2]
        )
        output_dict[detection_fields.detection_boxes] = (
            absolute_detection_boxlist.get()
        )
    detection_scores = detections[detection_fields.detection_scores][0]
    output_dict[detection_fields.detection_scores] = detection_scores

    if class_agnostic:
        detection_classes = tf.ones_like(detection_scores, dtype=tf.int64)
    else:
        detection_classes = (
            tf.to_int64(detections[detection_fields.detection_classes][0]) +
            label_id_offset
        )
    output_dict[detection_fields.detection_classes] = detection_classes

    if detection_fields.detection_masks in detections:
        detection_masks = detections[detection_fields.detection_masks][0]
        output_dict[detection_fields.detection_masks] = detection_masks
        if scale_to_absolute:
            # TODO: This should be done in model's postprocess
            # function ideally.
            detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_shape[1], image_shape[2]
            )
            detection_masks_reframed = tf.to_float(
                tf.greater(detection_masks_reframed, 0.5)
            )
            output_dict[detection_fields.detection_masks] = detection_masks_reframed

    if detection_fields.detection_keypoints in detections:
        detection_keypoints = detections[detection_fields.detection_keypoints][0]
        output_dict[detection_fields.detection_keypoints] = detection_keypoints
        if scale_to_absolute:
            absolute_detection_keypoints = keypoint_ops.scale(
                detection_keypoints, image_shape[1], image_shape[2]
            )
            output_dict[detection_fields.detection_keypoints] = (
                absolute_detection_keypoints
            )

    return output_dict

def visualize_detection_results(result_dict,
                                categories,
                                agnostic_mode=False,
                                min_score_thresh=.5,
                                max_num_predictions=20):
    if not set([
        'original_image', 'detection_boxes', 'detection_scores',
        'detection_classes'
    ]).issubset(set(result_dict.keys())):
        raise ValueError('result_dict does not contain all expected key.')
    logging.info('Creating detection visualizations.')
    category_index = label_map_util.create_category_index(categories)

    image = np.squeeze(result_dict['original_image'], axis=0)
    detection_boxes = result_dict['detection_boxes']
    detection_scores = result_dict['detection_scores']
    detection_classes = np.int32((result_dict['detection_classes']))
    detection_keypoints = result_dict.get('detection_keypoints', None)
    detection_masks = result_dict.get('detection_masks', None)

    print(detection_boxes)
    print(detection_classes)
    print(detection_scores)

    vis_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        instance_masks=detection_masks,
        keypoints=detection_keypoints,
        use_normalized_coordinates=False,
        max_boxes_to_draw=max_num_predictions,
        min_score_thresh=min_score_thresh,
        agnostic_mode=agnostic_mode
    )

    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    #with tf.gfile.Open("/Users/changetheworld/dev/git/tensorflow-rbnr/models/modelNumobj/eval/test.png", 'w') as fid:
    #    image_pil.save(fid, 'PNG')


    image_pil.show();
    #cv2.namedWindow('detection', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('detection', np.array(image_pil))
    #cv2.waitKey(0)

if __name__ == '__main__':
    tf.app.run()