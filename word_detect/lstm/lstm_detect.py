# Only workds for white background with black traces

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from skimage import transform
from validate import build_lstm, lstm_recognize

np.set_printoptions(threshold=np.nan)

def align_image(img):                                                                                                                                          
  # Reduce Background                                                                                                                                          
  sobel1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)                                                                                                         
  sobel2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)                                                                                                         
  sobel = np.abs(sobel1) + np.abs(sobel2)                                                                                                                      
  sobel = np.uint8(sobel)                                                                                                                                      
  ret, binary = cv2.threshold(sobel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)                                                                                

  # Cut Edges                                                                                                                                                  
  limit = 2 
  off = 4
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

def lstm_detect(img, network, limit_pixel=10, limit_char=60, target_h=300):
  # Readin Sentence Image, then Rescale to [32,32]
  img = align_image(img)

  scale = float(target_h) / img.shape[0]
  img = transform.rescale(img, scale)
  img[:3,:] = 1.0
  img[:,:3] = 1.0
  img[-3:,:] = 1.0
  img[:,-3:] = 1.0
  
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

  img_padded = np.pad(img, ((0,0), (limit_char,limit_char) ), 'constant', constant_values=(1.0))
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
            blank_region.append(tmp[0])
          blank_region.append(tmp[1])
            
        tmp = []
    
    if i == histogram.shape[0] - 1 and cont == True:
      if i - tmp[0] + 1 >= limit_char:
        tmp.append(i)
        blank_region.append(tmp[0])

  h = img.shape[0]
  
  # For Display Binary Image
  bin_show = cv2.cvtColor(binary_padded, cv2.COLOR_GRAY2BGR)

  # Generate words
  Color = [(255,0,0), (0,255,0), (0,0,255)]
  
  predicted_list = []
  for i in range(0, len(blank_region), 2):
    # Initialize a word
    off = 30
    left = blank_region[i] - off
    right = blank_region[i+1] + off
    
    delta = right - left + 1
    if delta <= 30:
      continue
    
    # Obtain word image
    word = img_padded[:, left:right + 1]
    
    # LSTM Network Inference
    image, width, sess, prediction = network
    pred_w = lstm_recognize(image, width, sess, prediction, word)
    
    # Record word
    predicted_list.append(pred_w)
    
    # Display Result
    cv2.rectangle(bin_show, (left, 0), (right, h-1), Color[1], 6)

  pred_sentence = " ".join(predicted_list)
  
  return pred_sentence, predicted_list
