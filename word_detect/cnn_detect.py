# Only workds for white background with black traces

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from skimage import transform
from net.resnet_train import resnet
from lstm.validate import build_lstm, lstm_recognize

np.set_printoptions(threshold=np.nan)

def build_cnn_network(model):
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

def cnn_recognize_word(image, score, sess, img):
  # Hash: label -> char
  hash_table = []
  for i in range(10):
    hash_table.append(str(i))
  for i in range(65, 91):
    hash_table.append(chr(i))
  for i in range(97, 123):
    hash_table.append(chr(i))

  img = transform.resize(img, [28, 28])
  if img.dtype == "float32" or img.dtype == "float64":
    img = np.uint8( (img * 255) )
  
  img = (255 - img).astype('uint8')
  
  img = np.expand_dims(img, axis=0)                                                                                                                      
  img = np.expand_dims(img, axis=-1)                                                                                                                     
  
  std = np.maximum(np.std(img, axis=(1,2,3)), 1.0/np.sqrt(img[0].size))                                                                                    
  mean = np.mean(img, axis=(1,2,3))
  img = ( (img.T - mean) / std).T
  
  class_score = sess.run(score, feed_dict={image: img})[0]

  prediction = np.argmax(class_score)
  pred_score = np.max(class_score)
  
  pred_char = hash_table[prediction]

  return pred_char, pred_score

def align_image(img):                                                                                                                                          
  # Reduce Background                                                                                                                                          
  sobel1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)                                                                                                         
  sobel2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)                                                                                                         
  sobel = np.abs(sobel1) + np.abs(sobel2)                                                                                                                      
  sobel = np.uint8(sobel)                                                                                                                                      
  ret, binary = cv2.threshold(sobel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)                                                                                
                                                                                                                                                               
  # Cut Edges                                                                                                                                                  
  limit = 2                                                                                                                                                    
  off = 2
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
  image = img[tb:bb+1,lb:rb+1]
  return image

if __name__ == "__main__":
  limit_pixel = 10
  limit_char = 5
  limit_word = 12 * limit_char
  target_h = 300
  path = "images/rand.png"

  # Readin Sentence Image, then Rescale to [32,32]
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = align_image(img)

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

  blanck_region_char = []
  blank_index = []
  blank_pos = []
  cont = False
  tmp = []

  # Perform Paddings
  img_padded = np.pad(img, ((0,0), (limit_char,limit_char) ), 'constant', constant_values=(0))
  sobel_padded = np.pad(sobel, ((0,0), (limit_char,limit_char) ), 'constant', constant_values=(0))
  binary_padded = np.pad(binary, ((0,0), (limit_char,limit_char) ), 'constant', constant_values=(0))
  histogram = np.pad(histogram, (limit_char,limit_char), 'constant', constant_values=(0))
  
  # Get Blank Regions in form of (start, end) 
  for i in range(histogram.shape[0]):
    if histogram[i] <= limit_pixel:
      if cont == False:
        tmp.append(i)
        cont = True
    else:
      if cont == True:
        cont = False
        blank_length = i - tmp[0]
        if blank_length >= limit_char:
          tmp.append(i - 1)
          if tmp[0] != 0:
            blanck_region_char.append(tmp[0])
          blanck_region_char.append(tmp[1])
            
        tmp = []
    
    if i == histogram.shape[0] - 1 and cont == True:
      if i - tmp[0] + 1 >= limit_char:
        tmp.append(i)
        blanck_region_char.append(tmp[0])

  h = img.shape[0]
  char_list = []
  
  # For Display Binary Image
  bin_show = cv2.cvtColor(binary_padded, cv2.COLOR_GRAY2BGR)

  # Generate words
  Color = [(255,0,0), (0,255,0), (0,0,255)]
  index = 0
  
  # Get Space Positions
  for i in range(1, len(blanck_region_char)-1, 2):
    left = blanck_region_char[i]
    right = blanck_region_char[i+1]

    if right - left >= limit_word:
      blank_index.append(float(left + right) / 2)
      ind = (i-1)/2
      blank_pos.append(ind)
      cv2.rectangle(bin_show, (left+15, 0), (right-15, h-1), Color[2], 6)
  
  # Recognize a word
  model_path = "model/"
  sess, graph_res, image, features = build_cnn_network(model_path)
  
  predicted_list = []
  for i in range(0, len(blanck_region_char), 2):
    # Initialize a word
    off = 2
    left = blanck_region_char[i] - off
    right = blanck_region_char[i+1] + off
    
    delta = right - left + 1
    if delta <= 30:
      continue
    
    # Obtain word image
    char = img_padded[:, left:right + 1]

    # Recognize Word
    pred_c, pred_s = cnn_recognize_word(image, features, sess, char)

    # Assembly Sentence
    predicted_list.append(pred_c)
    
    # Record Word
    char_list.append(char)

    # Display Result
    cv2.rectangle(bin_show, (left, 0), (right, h-1), Color[1], 6)
    index += 1
       
    cv2.imwrite("results/char_" + str(i) + ".jpg", char)
    
  # Assemble Sentence
  for i in range(len(blank_pos)):
    pos = blank_pos[i] + i
    predicted_list.insert(pos, " ")
  pred_sentence = "".join(predicted_list)
  print predicted_list
  print pred_sentence  

  plt.scatter(range(len(histogram)), histogram, marker=".")
  plt.show()
  
  img = np.uint8( (img * 255) )
  cv2.imwrite("results/cut.jpg", img)
  cv2.imwrite("results/sobel.jpg", sobel)
  cv2.imwrite("results/binary.jpg", binary)
  cv2.imwrite("results/divide.jpg", bin_show)
  
  cv2.imshow('test', bin_show)
  cv2.waitKey(0)
  
