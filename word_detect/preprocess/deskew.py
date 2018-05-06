""" Deskews file after getting skew angle """
import numpy as np
import sauvola_binarize
from detect_skew import SkewDetect
from skimage import io
from skimage.transform import rotate
import cv2


class Deskew:

    def __init__(self, img, display_image, output_file, r_angle):

        self.input_file = img
        self.display_image = display_image
        self.output_file = output_file
        self.r_angle = r_angle
        self.skew_obj = SkewDetect(img)

    def deskew(self):
        img = self.input_file
        res = self.skew_obj.process_single_file()
        angle = res['Estimated Angle']
        #print angle

        if angle >= 0 and angle <= 90:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -45 and angle < 0:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -90 and angle < -45:
            rot_angle = 90 + angle + self.r_angle

        #print rot_angle

        rotated = rotate(img, rot_angle, resize=True)

        if self.display_image:
            self.display(rotated)

        if self.output_file:
            self.saveImage(rotated*255)
        return rotated*255

    def saveImage(self, img):
        path = self.skew_obj.check_path(self.output_file)
        #io.imsave(path, img.astype(np.uint8))

    def display(self, img):

        cv2.imshow('result',img)
        cv2.waitKey(0)

    def run(self):

        if self.input_file is not None:
            img = self.deskew()
        return img


def run(img):
    deskew_obj = Deskew(
        img,
        None,
        'result.png',
        0)
    rotated = deskew_obj.run()
    return rotated
