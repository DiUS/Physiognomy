#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys

    fn = '/home/vwong/tmp/downloads/faces/cohn-kanade-images/S005/001/S005_001_00000011.png'

    frontalface_clf = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    mouth_clf       = cv2.CascadeClassifier('haarcascades/haarcascade_msc_mouth.xml')
    nose_clf        = cv2.CascadeClassifier('haarcascades/haarcascade_msc_nose.xml')

    img  = cv2.imread(fn)
    vis  = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # find all faces
    frontalfaces = detect(gray, frontalface_clf)
    draw_rects(vis, frontalfaces, (0, 255, 0))
    print(frontalfaces)

    for x1, y1, x2, y2 in frontalfaces:
        # TODO: increase/decrease sensitivity to find one and only one nose, mouth

        # assume nose is in the middle 1/3 of frontalface
        roi      = gray[y1+(y2-y1)/3:y2-(y2-y1)/3, x1+(x2-x1)/3:x2-(x2-x1)/3]
        vis_roi  =  vis[y1+(y2-y1)/3:y2-(y2-y1)/3, x1+(x2-x1)/3:x2-(x2-x1)/3]
        nose     = detect(roi.copy(), nose_clf)
        draw_rects(vis_roi, nose, (255, 0, 0))
        print(nose)

        # assume mouth in bottom 1/2 of frontalface
        roi      = gray[y1+(y2-y1)/2:y2, x1+(x2-x1)/2:x2]
        vis_roi  =  vis[y1+(y2-y1)/2:y2, x1+(x2-x1)/2:x2]
        mouth    = detect(roi.copy(), mouth_clf)
        draw_rects(vis_roi, mouth, (255, 0, 0))
        print(mouth)

    cv2.imwrite('foo.png', vis)
