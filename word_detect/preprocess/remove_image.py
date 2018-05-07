import numpy
# this package is used to do the wavelet transform
import pywt
from PIL import Image
import cv2
from matplotlib import pyplot as plt


# this is a function to read the image as gray image and convert it to gray background
def read_gray(directory):
    gray_image = numpy.array(Image.open(directory).convert("L"))
    height, width = gray_image.shape
    pixel_average = float(numpy.sum(gray_image)) / height / width
    if pixel_average > 128:
        # change the background to black
        gray_image = 255 - gray_image
    return gray_image


# the output of the function will be a fixed size image, ratio is defined as original/sampled
def down_sample(origin_image):
    height, width = origin_image.shape
    ratio = width / 1024.
    # if the original image is less than 1024 pixel in the width, then there is no need to reshape the image
    if ratio <= 1:
        return 1, origin_image
    height_output = height / ratio
    width_output = width / ratio
    # note that when using open-cv to resize the image, we have to place the width in the front
    return ratio, cv2.resize(origin_image, (int(width_output), int(height_output)))


# compose the 4 result image from wavelet transform into 1 image
def reconstruct(ll, lh, hl, hh):
    height, width = ll.shape
    result = numpy.zeros((2*height, 2*width))
    result[0:height, 0:width] = ll[0:height, 0:width]
    result[height:2*height, 0:width] = lh[0:height, 0:width]
    result[0:height, width:2*width] = hl[0:height, 0:width]
    result[height:2*height, width:2*width] = hh[0:height, 0:width]
    return result


# seed is used to find the points which have lots of possible neighbors
def seed(cdd, threshold=0.3):
    height, width = cdd.shape
    seed_m = numpy.zeros_like(cdd)
    pace_width = 5
    pace_height = 8
    points_num = (2 * pace_height) * (2 * pace_width)
    for i in range(pace_height, height - pace_height, pace_height):
        for j in range(pace_width, width - pace_width, pace_width):
            count = numpy.sum(cdd[i-pace_height:i+pace_height, j-pace_width:j+pace_width])
            if float(count) / points_num > threshold:
                seed_m[i - pace_height: i + pace_height, j - pace_width: j + pace_width] = 1
    return seed_m


# only display the image part which is not covered by the mask
def mask_image(image, masked):
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            image[i][j] = 0 if masked[i][j] < 128 else image[i][j]
    return image


# find the candidate pixels by comparing the energy in the high-pass filtered images
def find_candidate(image, lh, hl, hh):
    def energy(l_h, h_l, h_h):
        enr = numpy.zeros_like(l_h)
        block_height, block_width = h_l.shape
        for m in range(block_height):
            for n in range(block_width):
                enr[m][n] = numpy.sqrt(l_h[m][n] ** 2 + h_l[m][n] ** 2 + h_h[m][n] ** 2)
        return enr
    eng = energy(lh, hl, hh)
    hist_max = numpy.max(eng)
    hist, bin_edge = numpy.histogram(eng, bins=range(21), density=True)
    hist_pace = hist_max / 50.
    hist_threshold = 0
    hist_prob = 0
    for i in hist:
        hist_prob += i
        hist_threshold += hist_pace
        if hist_prob > 0.9:
            break

    result = numpy.zeros_like(image)
    height, width = result.shape
    for i in range(height // 2):
        for j in range(width // 2):
            result[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = 1 if eng[i][j] >= hist_threshold else 0
    return result


# define a mapping function to map each point in the sampled image to the original image
# ratio is defined as original/sampled
def mapping(point, ratio):
    if ratio == 1:
        return point
    return [[int(point[0][0] * ratio), int(point[0][1] * ratio)]]


# define a function to see the number of non-zero pixels in the detected blocks using the candidate image
def valued_block(cdd):
    count = numpy.sum(cdd)
    points = cdd.shape[0] * cdd.shape[1]
    if points == 0:
        return 0
    ratio = float(count) / points
    return ratio


def binarize(mask):
    result_mask = numpy.ones_like(mask)
    index_zero = numpy.where(mask == 0)
    result_mask[index_zero] = 0
    return result_mask

##################################
#   Begin of the main function   #
##################################
def remove_image(directory):
    original_image = read_gray(directory)
    # cv2.imwrite("./result_image/original_image.jpg", original_image)
    # save the original size of the image in order to recover later
    # origin_height, origin_width = original_image.shape
    # down-sample the image so that it can be processed in the same size
    sample_ratio, sample_image = down_sample(original_image)
    # sample_height, sample_width = sample_image.shape
    # cv2.imwrite("./result_image/down_sample.jpg", sample_image)

    # use wavelet transform the get 3 results of high-pass filters with different direction
    cA, (cH, cV, cD) = pywt.dwt2(sample_image, "haar")
    # candidate is the to find the pixel points which have the energy greater than the threshold
    # The image has been transformed into a binary image till now
    candidate = find_candidate(sample_image, cH, cV, cD)

    # cv2.imwrite("./result_image/candidate.jpg", candidate*255)
    # table lines can be removed in the future when blocks have been divided

    seed_matrix = seed(candidate)

    # cv2.imwrite("./result_image/seed.jpg", seed_matrix*255)
    dilate_kernel = numpy.ones((3, 7))
    # seed_matrix = cv2.dilate(seed_matrix, dilate_kernel)
    # do the closing operation so that the isolated part of the image will be removed
    close_seed_matrix = cv2.morphologyEx(seed_matrix, cv2.MORPH_CLOSE, dilate_kernel)
    # seed_matrix = cv2.dilate(seed_matrix, numpy.ones((3, 3)))
    # cv2.imwrite("./result_image/CloseSeed.jpg", close_seed_matrix*255)

    # draw the minimum contour of the rectangle
    ret, thresh = cv2.threshold(close_seed_matrix*255, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # mask is a binary image, indicating whether the pixel in the original image should be kept
    mask = numpy.zeros_like(original_image)

    for contour in contours:
        # if the area of the contour is less than a specific area, then this cannot be a real contour
        if cv2.contourArea(contour) < 2500:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        min_x, min_y = box.min(axis=0)
        max_x, max_y = box.max(axis=0)
        if max_y - min_y < 10 or max_x - min_x < 30:
            continue
        # print valued_block(seed_matrix[min_y:max_y, min_x:max_x])
        # points = (max_x-min_x) * (max_y - min_y)
        # ratio = numpy.sum(original_image[min_y:max_y, min_x:max_x] * seed_matrix[min_y:max_y, min_x:max_x]) / points
        # print ratio
        if valued_block(close_seed_matrix[min_y:max_y, min_x:max_x]) < .55:
            continue

        # map the contour to the original image
        map_contour = []
        for points in contour:
            map_point = mapping(points, sample_ratio)
            map_contour.append(map_point)
        map_contour = numpy.array(map_contour)
        im1 = cv2.drawContours(mask, [map_contour], 0, (255, 255, 255), -1)

    result_mask = binarize(mask)
    result_mask = cv2.dilate(result_mask, numpy.ones((5, 11)))
    result_mask = binarize(result_mask)

    # cv2.imwrite("./result_image/mask.jpg", result_mask * 255)
    # cv2.imwrite("./result_image/test_result.jpg", 255 - result_mask * original_image)

    # # plot the temporary image
    # plt.figure()
    # plt.subplot(2, 4, 1)
    # plt.imshow(original_image, cmap=plt.cm.gray)
    # plt.title("Origin")
    # plt.axis("off")
    #
    # plt.subplot(2, 4, 2)
    # plt.imshow(sample_image, cmap=plt.cm.gray)
    # plt.title("Down-sample")
    # plt.axis("off")
    #
    # plt.subplot(2, 4, 3)
    # plt.imshow(candidate * 255, cmap=plt.cm.gray)
    # plt.title("Candidate")
    # plt.axis("off")
    #
    # plt.subplot(2, 4, 4)
    # plt.imshow(seed_matrix * 255, cmap=plt.cm.gray)
    # plt.title("seed")
    # plt.axis("off")
    # ,
    # ,
    # ,
    # ,
    # close_seed_matrix * 255,
    # result_mask * 255,
    # result_mask * original_image

    return result_mask * original_image
