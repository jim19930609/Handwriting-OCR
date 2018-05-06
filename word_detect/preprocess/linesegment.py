import cv2
import numpy as np
import sauvola_binarize
import deskew

filenames = ['testslant1.jpg']

## define function to localize line
def locateline(im, thresh):
    # after transformation black pixel has value 1, white pixel has value 0
    im = abs(im - 255.0)
    im = im / 255

    (h,w) = np.shape(im)
    linehist = [0 for i in range(h)]
    for i in range(h):
        row = im[i][:]
        sum = np.sum(row)
        linehist[i] = sum

    # blankindex store the index of row for each blank area
    # the length of blankindex is the number of blank area
    blankindex = []
    index = []
    for j in range(h):
        if j == h - 1 and len(index) > 0:
            blankindex.append(index)
            index = []
        elif linehist[j] <= thresh:
            index.append(j)
        elif linehist[j] > thresh and len(index) > 0:
            blankindex.append(index)
            index = []
        else:
            continue

    # lineindex store the index of each line of a certain part of image
    # lineindexes: each element store the index of each line of a certain part of image, image is divided by big blank areas
    lineindexes = []
    lineindex = []
    for i in range(len(blankindex)):
        if len(blankindex[i]) > 30 and i == 0:
            index = blankindex[i][len(blankindex[i]) - 3]
            lineindex.append(index)
        elif len(blankindex[i]) > 30:
            index = blankindex[i][3]
            lineindex.append(index)
            lineindexes.append(lineindex)
            lineindex = []
            index = blankindex[i][len(blankindex[i])-3]
            lineindex.append(index)
        elif len(blankindex[i]) < 30 and i == len(blankindex) - 1:
            index = (blankindex[i][0] + blankindex[i][len(blankindex[i]) - 1]) / 2
            lineindex.append(index)
            lineindexes.append(lineindex)
        elif len(blankindex[i]) < 30:
            index = (blankindex[i][0] + blankindex[i][len(blankindex[i]) - 1]) / 2
            lineindex.append(index)

    #print lineindexes
    return lineindexes

def lineseg(lineindexes, im):
  (h,w) = np.shape(im)
  sentence_list = []
  for i in range(len(lineindexes)):
      for j in range(len(lineindexes[i])-1):
          # ignore long straight line in document
          if lineindexes[i][j+1] - lineindexes[i][j] < 10:
              continue
          else:
              x1 = lineindexes[i][j]
              x2 = lineindexes[i][j+1]
              y1 = 0
              y2 = w-1
              crop = im[x1:x2, y1:y2]
              sentence_list.append(crop)


  return sentence_list

def Line_separation(img):
  imbw = 255 - sauvola_binarize.binarize(img)
  #print 'finish binarize'
  
  kernel = np.ones((3,3),np.uint8)
  dialation_image = cv2.dilate(imbw,kernel,iterations=1)
  blur_image = cv2.blur(dialation_image,(3,3))
  rotated_img = 255 - deskew.run(blur_image)
  
  # image used to do line segment must have white back and black words
  lineindex =  locateline(rotated_img,0)
  sentence_list = lineseg(lineindex, rotated_img)
  
  #print 'finish line segmentation'
  return sentence_list
  

