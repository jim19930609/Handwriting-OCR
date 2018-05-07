import numpy
from PIL import Image
import wavelet
import gabor
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


def main(directory, method="gabor"):
    image = read_gray(directory)
    if method == "gabor":
        return gabor.gabor_image_removal(image)
    elif method == "wavelet":
        return wavelet.wavelet_image_removal(image)


# if __name__ == "__main__":
#     dirc = "./source_image/final.jpg"
#     result1 = main(dirc)
#     result2 = main(dirc, "wavelet")
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(result1, cmap=plt.cm.gray)
#     plt.axis("off")
#     plt.title("Gabor")
#     plt.subplot(122)
#     plt.imshow(result2, cmap=plt.cm.gray)
#     plt.axis("off")
#     plt.title("Wavelet")
#     plt.show()