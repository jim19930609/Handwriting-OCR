import numpy
import pywt
import cv2


# seed is used to find the points which have lots of possible neighbors
def seed(cdd, threshold=0.35):
    height, width = cdd.shape
    seed_m = numpy.zeros_like(cdd)
    pace_width = 40
    pace_height = 15
    points_num = (2 * pace_height) * (2 * pace_width)
    for i in range(pace_height, height - pace_height, 4):
        for j in range(pace_width, width - pace_width, 4):
            count = numpy.sum(cdd[i-pace_height:i+pace_height, j-pace_width:j+pace_width])
            if float(count) / points_num > threshold:
                seed_m[i - pace_height: i + pace_height, j - pace_width: j + pace_width] = 1
    return seed_m


# find the candidate pixels by comparing the energy in the high-pass filtered images
def find_candidate(image, lh, hl, hh):
    def energy(l_h, h_l, h_h):
        enr = numpy.zeros_like(l_h)
        block_height, block_width = h_l.shape
        for m in range(block_height):
            for n in range(block_width):
                enr[m][n] = numpy.sqrt(l_h[m][n] ** 2 + h_l[m][n] ** 2 + 4 * h_h[m][n] ** 2)
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


def binarize(mask):
    result_mask = numpy.ones_like(mask)
    index_zero = numpy.where(mask == 0)
    result_mask[index_zero] = 0
    return result_mask

##################################
#   Begin of the main function   #
##################################
def wavelet_image_removal(original_image):
    # use wavelet transform the get 3 results of high-pass filters with different direction
    cA, (cH, cV, cD) = pywt.dwt2(original_image, "haar")
    # The image has been transformed into a binary image till now
    candidate = find_candidate(original_image, cH, cV, cD)
    # table lines can be removed in the future when blocks have been divided

    seed_matrix = seed(candidate)

    dilate_kernel = numpy.ones((3, 11))
    # seed_matrix = cv2.dilate(seed_matrix, dilate_kernel)
    # do the closing operation so that the isolated part of the image will be removed
    close_seed_matrix = cv2.morphologyEx(seed_matrix, cv2.MORPH_CLOSE, dilate_kernel)
    # seed_matrix = cv2.dilate(seed_matrix, numpy.ones((3, 3)))
    close_seed_matrix = binarize(close_seed_matrix)

    return 255 - close_seed_matrix * original_image
