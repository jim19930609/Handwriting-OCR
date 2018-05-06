import cv2
import sauvola

def binarize(im):

    #print '- read image..'
    #im = cv2.imread(filename, 0)

    # pure sauvola algorithm
    #print '- binarize image..'
    # imbw = sauvola.binarize(im, [5, 5], 100, 0.8)
    imbw = sauvola.binarize(im, [50, 50], 128, 0.5)
    # cv2.imwrite('/Users/yinanwang/PycharmProject/dip_courseproj/sauvola_result.png', imbw)

    return imbw

