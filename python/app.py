#!/usr/bin/python
import facedetect
import gabor_filter
import csv

if __name__ == '__main__':
    import sys

    training_set = []
    with open('./dataset_prep/classified_images.csv','rb') as csvfile:
      file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in file_reader:
        result = facedetect.detect_faces(row[0])
        if len(result['face']) != 1:
            next
        face = result['face'][0]
        face_vector = gabor_filter.filter_face(row[0],face[0], face[1], face[2], face[3])
        print row[1]
        print len(face_vector)
        print face_vector


