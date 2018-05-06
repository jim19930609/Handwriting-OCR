# CNN-LSTM-CTC-OCR
# Copyright (C) 2017, 2018 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
from skimage import transform,data
import cv2
import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import learn

import mjsynth
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model','lstm/model',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('device','/gpu:0',
                           """Device for graph placement""")

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers

def _get_image(filename):
    """Load image data for placement in graph"""
    image = Image.open(filename) 
    image = np.array(image)
    # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
    if image.ndim == 3:
        image = image[:,:,:1] # so just extract first channel, preserving 3D shape
    else:
        image = np.expand_dims(image, axis=-1)
    return image


def _preprocess_image(image):

    # Copied from mjsynth.py. Should be abstracted to a more general module.
    
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    image = tf.concat([first_row, image], 0)

    return image


def _get_input():
    """Set up and return image and width placeholder tensors"""

    # Raw image as placeholder to be fed one-by-one by dictionary
    image = tf.placeholder(tf.uint8, shape=[31, None, 1])
    width = tf.placeholder(tf.int32,shape=[]) # for ctc_loss

    return image,width


def _get_output(rnn_logits,sequence_length):
    """Create ops for validation
       predictions: Results of CTC beacm search decoding
    """
    with tf.name_scope("test"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=True)

    return predictions


def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config


def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path


def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn

def _get_string(labels):
    """Transform an 1D array of labels into the corresponding character string"""
    string = ''.join([mjsynth.out_charset[c] for c in labels])
    return string

def align_image(img):
  # Reduce Background
  sobel1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
  sobel2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
  sobel = np.abs(sobel1) + np.abs(sobel2)
  sobel = np.uint8(sobel)
  ret, binary = cv2.threshold(sobel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
  
  # Cut Edges
  limit = 2
  off = 5
  lb = 0
  rb = 0
  tb = 0
  bb = 0
  
  for j in range(binary.shape[1]):
    hist = np.sum(binary[:,j])
    if hist >= limit:
      lb = j
      if lb >= off:
        lb -= off
      else:
        lb = 0
      break

  for j in range(binary.shape[1])[::-1]:
    hist = np.sum(binary[:,j])
    if hist >= limit:
      rb = j
      if binary.shape[1]-rb > off:
        rb += off
      else:
        rb = binary.shape[1] - 1
      break

  for i in range(binary.shape[0]):
    hist = np.sum(binary[i,:])
    if hist >= limit:
      tb = i
      if tb >= off:
        tb -= off
      else:
        tb = 0
      break

  for i in range(binary.shape[0])[::-1]:
    hist = np.sum(binary[i,:])
    if hist >= limit:
      bb = i
      if binary.shape[0]-bb > off:
        bb += off
      else:
        bb = binary.shape[0] - 1
      break
  
  image = img[tb:bb+1, lb:rb+1]
  
  # Transform and Resize
  scale = float(31.0/image.shape[0])                                                                                                            
  image = transform.rescale(image, [scale,scale]) * 255
  image = np.uint8(image)

  
  image_tmp = image[:,:,0]
  image_tmp = np.pad(image_tmp, ((0,0), (5,5) ), 'constant', constant_values=(255))
  image = np.expand_dims(image_tmp, axis=-1)
  
  return image

def build_lstm():
  graph = tf.Graph()
  with graph.as_default():
    image,width = _get_input() # Placeholder tensors

    proc_image = _preprocess_image(image)
    proc_image = tf.reshape(proc_image,[1,32,-1,1]) # Make first dim batch

    with tf.device(FLAGS.device):
        features,sequence_length = model.convnet_layers( proc_image, width, 
                                                         mode)
        logits = model.rnn_layers( features, sequence_length,
                                   mjsynth.num_classes() )
        prediction = _get_output( logits,sequence_length)

    saver = tf.train.Saver()
    session_config = _get_session_config()
    
    init_op = tf.group( tf.global_variables_initializer(),
                        tf.local_variables_initializer()) 
  
  sess = tf.Session(config=session_config, graph=graph)  
  sess.run(init_op)
  model_dir = "lstm/model/model.ckpt-81950"
  saver.restore(sess, model_dir)

  return image, width, sess, prediction

def lstm_recognize(image, width, sess, prediction, img):
  if img.dtype == "float32" or img.dtype == "float64":
    img = np.uint8( (img * 255) )
  
  image_data = np.expand_dims(img, axis=-1)
  image_data = align_image(image_data)
  
  #Get prediction for single image (isa SparseTensorValue)
  [output] = sess.run(prediction,{ image: image_data, 
                                   width: image_data.shape[1]} )
  pred = _get_string(output.values)
  return pred

if __name__ == '__main__':
    tf.app.run()
