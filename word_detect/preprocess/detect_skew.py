""" Calculates skew angle """
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import cv2


class SkewDetect:

    piby4 = np.pi / 4

    def __init__(
        self,
        input_file=None,
        batch_path=None,
        output_file=None,
        sigma=3.0,
        display_output=None,
        num_peaks=20,
        plot_hough=None
    ):

        self.sigma = sigma
        self.input_file = input_file
        self.batch_path = batch_path
        self.output_file = output_file
        self.display_output = display_output
        self.num_peaks = num_peaks
        self.plot_hough = plot_hough

    def write_to_file(self, wfile, data):

        for d in data:
            wfile.write(d + ': ' + str(data[d]) + '\n')
        wfile.write('\n')

    def get_max_freq_elem(self, arr):

        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)

        return max_arr

    def display_hough(self, h, a, d):

        plt.imshow(
            np.log(1 + h),
            extent=[np.rad2deg(a[-1]), np.rad2deg(a[0]), d[-1], d[0]],
            cmap=plt.cm.gray,
            aspect=1.0 / 90)
        plt.show()

    def compare_sum(self, value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False

    def display(self, data):
        
        for i in data:
            print i + ": " + str(data[i])

    def calculate_deviation(self, angle):

        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)

        return deviation

    def run(self):

        if self.display_output:
            if self.display_output.lower() == 'yes':
                self.display_output = True
            else:
                self.display_output = False

        if self.plot_hough:
            if self.plot_hough.lower() == 'yes':
                self.plot_hough = True
            else:
                self.plot_hough = False

        if self.input_file is None:
            if self.batch_path:
                self.batch_process()
            else:
                print "Invalid input, nothing to process."
        else:
            print 'single'
            self.process_single_file()

    def check_path(self, path):

        if os.path.isabs(path):
            full_path = path
        else:
            full_path = os.getcwd() + '/' + str(path)
        return full_path

    def process_single_file(self):

        # file_path = self.check_path(self.input_file)
        res = self.determine_skew()

        if self.output_file:
            output_path = self.check_path(self.output_file)
            wfile = open(output_path, 'w')
            self.write_to_file(wfile, res)
            wfile.close()
        #print res
        return res


    def determine_skew(self):

        # img = io.imread(img_file, as_grey=True)
        img = self.input_file
        edges = canny(img, sigma=self.sigma)
        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)

        if len(ap) == 0:
            return {"Image File": img, "Message": "Bad Quality"}

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:

            deviation_sum = int(90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = int(ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = int(-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = int(90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            l = len(angles[j])
            if l > lmax:
                lmax = l
                maxi = j

        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)

        else:
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        data = {
            "Image File": 'img',
            "Average Deviation from pi/4": average_deviation,
            "Estimated Angle": ans_res,
            "Angle bins": angles}
        '''
        if self.display_output:
            self.display(data)
        '''

        if self.plot_hough:
            self.display_hough(h, a, d)
        return data
