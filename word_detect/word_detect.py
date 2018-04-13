import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from skimage import transform
from net.resnet_train import resnet

np.set_printoptions(threshold=np.nan)

hash_table = []
for i in range(10):
  hash_table.append(str(i))
for i in range(65, 91):
  hash_table.append(chr(i))
for i in range(97, 123):
  hash_table.append(chr(i))

def build_network(model):
  graph_res = tf.Graph()
  with graph_res.as_default():
    image = tf.placeholder(tf.float32, [None, 28, 28, 1])
    
    _,fc = resnet(image, 56, 62)
    score = tf.nn.softmax(fc)
    prediction = tf.argmax(score, 1)

    sess = tf.Session(graph=graph_res)
    
    saver = tf.train.Saver()
    saver.restore(sess, model)

  return sess, graph_res, image, score


def recognize_word(image, score, sess, img):
  img = np.expand_dims(img, axis=0)                                                                                                                      
  img = np.expand_dims(img, axis=-1)                                                                                                                     
  
  std = np.maximum(np.std(img, axis=(1,2,3)), 1.0/np.sqrt(img[0].size))                                                                                    
  mean = np.mean(img, axis=(1,2,3))
  img = ( (img.T - mean) / std).T
  
  class_score = sess.run(score, feed_dict={image: img})[0]

  prediction = np.argmax(class_score)
  pred_score = np.max(class_score)

  return prediction, pred_score


if __name__ == "__main__":
  range_limit = 5
  target_h = 300
  path = "images/test.png"

  # Readin Sentence Image, then Rescale to [32,32]
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  scale = float(target_h) / img.shape[0]
  img = transform.rescale(img, scale)

  # Generate binary image
  sobel1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 7)
  sobel2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 7)
  sobel = np.abs(sobel1) + np.abs(sobel2)
  sobel = np.uint8(sobel)
  ret, binary = cv2.threshold(sobel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

  # Compute Histogram
  histogram = np.sum(binary, axis = 0) / 255

  blank_region = []
  cont = False
  tmp = []

  # Perform Paddings
  img_padded = np.pad(img, (range_limit, range_limit ), 'constant', constant_values=(0, 0))
  sobel_padded = np.pad(sobel, (range_limit, range_limit ), 'constant', constant_values=(0, 0))
  binary_padded = np.pad(binary, (range_limit, range_limit ), 'constant', constant_values=(0, 0))
  histogram = np.pad(histogram, (range_limit, range_limit ), 'constant', constant_values=(0, 0))

  # Get Blank Regions in form of (start, end) 
  for i in range(histogram.shape[0]):
    if histogram[i] <= 10:
      if cont == False:
        tmp.append(i)
        cont = True
    else:
      if cont == True:
        cont = False
        if i - tmp[0] >= range_limit:
          tmp.append(i - 1)
          if tmp[0] != 0:
            blank_region.append(tmp[0])
          blank_region.append(tmp[1])
        tmp = []
    
    if i == histogram.shape[0] - 1 and cont == True:
      if i - tmp[0] + 1 >= range_limit:
        tmp.append(i)
        blank_region.append(tmp[0])

  h = img.shape[0]
  word_list = []
  
  # For Display Binary Image
  bin_show = cv2.cvtColor(binary_padded, cv2.COLOR_GRAY2BGR)

  # Generate words
  Color = [(255,0,0), (0,255,0)]
  index = 0
  
  # Recognize a word
  model_path = "model/"
  sess, graph_res, image, features = build_network(model_path)
  
  for i in range(0, len(blank_region), 2):
    # Initialize a word
    off = 2
    left = blank_region[i] - off
    right = blank_region[i+1] + off
    
    delta = right - left + 1
    if delta <= 30:
      continue
    
    # Obtain word image
    word = img_padded[:, left:right + 1]
    word = transform.resize(word, [28, 28])
    
    # Recognize Word
    pred_c, pred_s = recognize_word(image, features, sess, word)
    print pred_c, hash_table[pred_c], pred_s
    
    # Record Word
    word_list.append(word)

    # Display Result
    cv2.rectangle(bin_show, (left, 0), (right, h-1), Color[index % 2], 2)
    index += 1
    
    cv2.imshow('test', word)
    cv2.waitKey(0)

  cv2.imshow('test', bin_show)
  cv2.waitKey(0)
