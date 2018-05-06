import numpy as np
import tensorflow as tf
import os
def one_hot_vec(label, classes):
    vec = np.zeros(classes)
    vec[label] = 1
    return vec

def test_inputs(data_list_path,
                 batch_size,
                 num_buffer=8,
                 num_preprocess_threads=None):
    
    def get_image_paths_and_labels(list_path):
        image_paths_flat = []
        labels_flat = []
        for line in open(os.path.expanduser(list_path), 'r'):
            part_line = line.split(' ')
            image_path = part_line[0]
            label_index = part_line[1]
            image_paths_flat.append(image_path)
            labels_flat.append(int(label_index))

        return image_paths_flat, labels_flat
 
    def _parse_function(filename, label):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_jpeg(file_contents, channels=num_channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.per_image_standardization(image)

        return image, label

    def _gen():
        #rand_idx = np.random.permutation(num_examples)
        #for idx in rand_idx:
        for idx in range(num_examples):
            yield (image_list[idx], one_hot_vec(label_list[idx], num_classes))

    num_channels = 1
    image_list, label_list = get_image_paths_and_labels(data_list_path)
    num_classes = max(label_list) + 1
    num_examples = len(image_list)
    
    dataset = tf.data.Dataset.from_generator(_gen, (tf.string, tf.int32))
    dataset = dataset.map(_parse_function, num_parallel_calls=num_preprocess_threads)
    dataset = dataset.prefetch(buffer_size=num_buffer * batch_size)
    dataset = dataset.batch(batch_size)
    image_batch, label_batch = dataset.make_one_shot_iterator().get_next()

    return image_batch, label_batch, num_classes, num_examples

def train_inputs(data_list_path,
                 batch_size,
                 num_repeat,
                 num_buffer=4,
                 num_preprocess_threads=None):
    
    def get_image_paths_and_labels(list_path):
        image_paths_flat = []
        labels_flat = []
        for line in open(os.path.expanduser(list_path), 'r'):
            part_line = line.split(' ')
            image_path = part_line[0]
            label_index = part_line[1]
            image_paths_flat.append(image_path)
            labels_flat.append(int(label_index))

        return image_paths_flat, labels_flat
 
    def _parse_function(filename, label):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_jpeg(file_contents, channels=num_channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.per_image_standardization(image)

        return image, label

    def _gen():
        rand_idx = np.random.permutation(num_examples)
        for idx in rand_idx:
        #for idx in range(num_examples):
            yield (image_list[idx], one_hot_vec(label_list[idx], num_classes))

    num_channels = 1
    image_list, label_list = get_image_paths_and_labels(data_list_path)
    num_classes = max(label_list) + 1
    num_examples = len(image_list)
    
    dataset = tf.data.Dataset.from_generator(_gen, (tf.string, tf.int32))
    dataset = dataset.map(_parse_function, num_parallel_calls=num_preprocess_threads)
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.prefetch(buffer_size=num_buffer * batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_repeat)
    image_batch, label_batch = dataset.make_one_shot_iterator().get_next()

    return image_batch, label_batch, num_classes, num_examples

