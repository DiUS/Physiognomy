#!/usr/bin/python
import cv2.cv as cv
import cv2
import numpy as np
import math
from optparse import OptionParser
image_scale = 1

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
      fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
      np.maximum(accum, fimg, accum)
    return accum

def filter_face(input_name, x,y,w,h):
    filters = build_filters()
    img  = cv2.imread(input_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cropped = gray[y:h, x:w]
    img_arr = np.asarray(cropped[:,:])
    new_img = process(img_arr, filters)
    cv2.imshow('edge detection',new_img)
    return new_img

if __name__ == '__main__':
    parser = OptionParser(usage = "usage: %prog [options] [filename|x1,x2,y1,y2")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = "/usr/local/Cellar/opencv/2.4.8.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    (options, args) = parser.parse_args()
    input_name = args[0]
    x = int(args[1])
    y = int(args[2])
    w = int(args[3])
    h = int(args[4])
    print filter_face(input_name, x, y, w, h)
    cv.WaitKey(0)

    cv.DestroyWindow("result")

