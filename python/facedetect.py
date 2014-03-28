#!/usr/bin/env python

import cv2
import cv2.cv as cv
import csv
import os

# https://stackoverflow.com/questions/20801015/opencv-detectmultiscale-parameters

# scaleFactor - Parameter specifying how much the image size is reduced at each image scale.
# Basically the scale factor is used to create your scale pyramid. More explanation can be found here. 1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5 %, you increase the chance of a matching size with the model for detection is found.

# minNeighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.
# This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.

# minSize - Minimum possible object size. Objects smaller than that are ignored.
# This parameter determine how small size you want to detect. You decide it! Usually, [30, 30] is a good start for face detections.

# maxSize - Maximum possible object size. Objects bigger than that are ignored.
# This parameter determine how big size you want to detect. Again, you decide it! Usually, you don't need to set it manually, which means you want to detect any big, i.e. don't want to miss any one that is big enough.

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.02, minNeighbors=3, minSize=(30, 30), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def detect_faces(filename):
    frontalface_clf = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    eye_clf         = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    mouth_clf       = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

    img  = cv2.imread(filename)
    vis  = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # find all faces, then keep the largest
    frontalfaces = detect(gray, frontalface_clf)
    maxWidth = 0
    for x1, y1, x2, y2 in frontalfaces:
        if x2-x1 > maxWidth:
            maxWidth = x2-x1
            face = [[x1, y1, x2, y2]]
    draw_rects(vis, face, (0, 255, 0))
    print(face)

    # find features within largest face
    for x1, y1, x2, y2 in face:
        # eyes are in top 1/2 of face
        w        = x2-x1
        h        = y2-y1
        roi      = gray[y1+0.2*h:y2-0.5*h, x1+0.1*w:x2-0.1*w]
        vis_roi  =  vis[y1+0.2*h:y2-0.5*h, x1+0.1*w:x2-0.1*w]
        eye      = detect(roi.copy(), eye_clf)
        draw_rects(vis_roi, eye, (0, 0, 255))
        print(eye)

        # mouth is in bottom 1/3 of face
        roi      = gray[y1+0.7*h:y2-0.1*h, x1+0.2*w:x2-0.2*w]
        vis_roi  =  vis[y1+0.7*h:y2-0.1*h, x1+0.2*w:x2-0.2*w]
        mouth    = detect(roi.copy(), mouth_clf)
        draw_rects(vis_roi, mouth, (255, 0, 0))
        print(mouth)

    cv2.imwrite("analysed/" + os.path.basename(filename), vis)
    #cv2.waitKey(0);

if __name__ == '__main__':
    import sys
    
    with open('./dataset_prep/classified_images.csv','rb') as csvfile:
      file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in file_reader:
        detect_faces(row[0])
