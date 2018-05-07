import numpy
from PIL import Image
import cv2
from sklearn.cluster import KMeans


# this is a function to read the image as gray image and convert it to gray background
def read_gray(directory):
    gray_image = numpy.array(Image.open(directory).convert("L"))
    height, width = gray_image.shape
    pixel_average = float(numpy.sum(gray_image)) / height / width
    if pixel_average > 128:
        # change the background to black
        gray_image = 255 - gray_image
    return gray_image


# seed is used to find the points which have lots of possible neighbors
def seed(cdd, threshold=0.6):
    height, width = cdd.shape
    seed_m = numpy.zeros_like(cdd)
    pace_width = 35
    pace_height = 20
    points_num = (2 * pace_height) * (2 * pace_width)
    for i in range(pace_height, height - pace_height, 4):
        for j in range(pace_width, width - pace_width, 4):
            count = numpy.sum(cdd[i-pace_height:i+pace_height, j-pace_width:j+pace_width])
            if float(count) / points_num > threshold:
                seed_m[i - pace_height: i + pace_height, j - pace_width: j + pace_width] = 1
    return seed_m


def binarize(mask):
    result_mask = numpy.ones_like(mask)
    index_zero = numpy.where(mask == 0)
    result_mask[index_zero] = 0
    return result_mask


##################################
#   Begin of the main function   #
##################################
def gabor_image_removal(original_image):
    height = original_image.shape[0]
    width = original_image.shape[1]
    size = height * width
    shape = (int(height / 100), int(width / 100))
    kernels = []

    # M is the number of detected angles
    M = 8
    # N is the number of different frequency
    N = 3
    thetas = [i * numpy.pi / M for i in range(M)]
    f = [0.75 * numpy.pi / (2 ** i) for i in range(N)]
    lbd = [2 * numpy.pi / f_i for f_i in f]
    c_1 = 3
    c_2 = 16 / numpy.pi
    gamma = c_1 / c_2
    # sigma is used to normalize the covariance, and has the same length as the lbd
    sigma = [c_1 / f_i for f_i in f]
    for n in range(N):
        for m in range(M):
            # sigma is the standard deviation of the gaussian envelop
            # theta is the orientation of the normal to the parallel stripes of a Gabor Filter
            # lambda is the wavelength of the sinusoidal factor
            # gamma is the spatial aspect ratio
            # psi is the phase offset, default is pi*0.5
            GaborKernel = cv2.getGaborKernel(shape, sigma=sigma[n], theta=thetas[m], lambd=lbd[n], gamma=gamma, psi=0,
                                             ktype=cv2.CV_32F)
            GaborKernel *= 2 * numpy.pi * (sigma[n] ** 2) / gamma
            kernels.append(GaborKernel)

    # filtered_image = []
    avg_kernel_length = 5
    avg_kernel = numpy.ones((avg_kernel_length, avg_kernel_length)) / (avg_kernel_length ** 2)
    flt_img = numpy.zeros((height, width, M * N), dtype=float)
    for i in range(len(kernels)):
        filter_image = cv2.filter2D(original_image, -1, kernels[i])
        filter_image = numpy.tanh(filter_image * 0.25)
        energy = cv2.filter2D(filter_image, -1, avg_kernel)
        flt_img[:, :, i] = energy

    # plot the temporary feature result
    num = M * N
    row = (num - 1) // 8 + 1

    # only sampled 1/10 of the points to minimize the computation load
    cluster_num = 3
    sample_index = numpy.random.choice(size, size / 100, replace=False)
    sample = [[flt_img[sample_index[j] // width][sample_index[j] % width][i] for i in range(M * N)] for j in
              range(len(sample_index))]
    sample = numpy.array(sample)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(sample)
    # print kmeans.labels_

    mask_image = numpy.zeros((cluster_num, height, width), dtype=float)
    for i in range(height):
        feature = flt_img[i]
        cluster = kmeans.predict(feature)
        for j in range(width):
            mask_image[cluster[j]][i][j] = 1

    # normally, the text cluster will have the middle sum value, so, this is selecting the text mask
    sum_value = [numpy.sum(mask_image[i]) for i in range(len(mask_image))]
    arg_sort = numpy.argsort(sum_value)
    candidate = mask_image[arg_sort[1]]

    # filtrate the candidate
    seed_matrix = seed(candidate)

    # closing half distinguishable character
    close_kernel = numpy.ones((11, 19))
    # do the closing operation so that the isolated part of the image will be removed
    close_seed_matrix = cv2.morphologyEx(seed_matrix, cv2.MORPH_CLOSE, close_kernel)

    result_mask = binarize(close_seed_matrix)
    return 255 - result_mask * original_image
