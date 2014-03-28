#!/usr/bin/env python
import facedetect
import gabor_filter
import csv
import os
import sys, math
from PIL import Image
import align_faces
from sklearn import svm

if __name__ == '__main__':
    import sys

    X = []
    y = []
    with open('./dataset_prep/classified_images.csv','rb') as csvfile:
      file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in file_reader:
        result = facedetect.detect_faces(row[0])

        if len(result['face']) != 1:
            next

        face = result['face'][0]
        aligned_filename = "gabor_aligned/" + os.path.basename(row[0])
        align_faces.CropFace(Image.open(row[0]), eye_left=(face[0],face[1]), eye_right=(face[2],face[1]), offset_pct=(0.08,0.08), dest_sz=(200,200)).save(aligned_filename)
        face_vector = gabor_filter.filter_face(aligned_filename,0, 0, 200, 200)
        X.append(face_vector)
        y.append(row[1])

    clf = svm.SVC()
    clf.fit(X,y)
    result = facedetect.detect_faces("cohn-kanade-images/S005/001/S005_001_00000006.png")
    face = result['face'][0]
    aligned_filename = "gabor_aligned/" + os.path.basename(row[0])
    align_faces.CropFace(Image.open(row[0]), eye_left=(face[0],face[1]), eye_right=(face[2],face[1]), offset_pct=(0.08,0.08), dest_sz=(200,200)).save(aligned_filename)
    face_vector = gabor_filter.filter_face(aligned_filename,0, 0, 200, 200)

    labels = clf.predict([face_vector])
    print (labels[0])


