import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)

import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings('ignore') # Suppress Matplotlib warnings

from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def main(path_to_ckpt, path_to_cfg, data_dir, detection_dir, test_set='test'):
    print('Loading model... ', end='')
    start_time = time.time()
    
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(path_to_cfg)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(path_to_ckpt).expect_partial()
    
    @tf.function
    def detect_fn(image):
        """Detect objects in image.
        """
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    
    path_to_labels = os.path.join(data_dir, 'label_map.pbtxt')
    category_index = label_map_util.create_category_index_from_labelmap(
        path_to_labels, use_display_name=True)
    
    # Get image_paths
    examples_path = os.path.join(data_dir, 'sets', test_set+'.txt')
    examples_list = open(examples_path, 'r').read().split('\n')
    while '' in examples_list:
        examples_list.remove('')
    
    image_path = data_dir+'/images/%s.jpg'
    images = []
    for example in examples_list:
        images.append(image_path % example)
    
    os.makedirs(detection_dir, exist_ok=True)
    
    # Running inference
    for image_path in images:
        print(f'Running inference for {image_path}... ', end='')
        image_np = load_image_into_numpy_array(image_path)
        
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections[
            'detection_classes'].astype(np.int64)
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)
        
        name, extension = os.path.splitext(os.path.basename(image_path))
        
        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.savefig(os.path.join(detection_dir, name + extension), dpi=300)
        
        pkl = open(os.path.join(detection_dir, name + '.pkl'), 'wb')
        pickle.dump(detections, pkl)
        pkl.close()
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ckpt')
    parser.add_argument('--path_to_cfg')
    parser.add_argument('--data_dir')
    parser.add_argument('--test_set')
    parser.add_argument('--detection_dir')
    args = parser.parse_args()
    
    main(args.path_to_ckpt, args.path_to_cfg, args.data_dir,
         args.detection_dir, args.test_set)
