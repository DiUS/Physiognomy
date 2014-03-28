#!/usr/bin/python
import facedetect
import gabor_filter
import csv

if __name__ == '__main__':
    import sys

    with open('./dataset_prep/classified_images.csv','rb') as csvfile:
      file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in file_reader:
        result = facedetect.detect_faces(row[0])
        print(result['face'])
        if len(result['face']) != 1:
            next
        face = result['face'][0]
        print gabor_filter.filter_face(row[0],face[0], face[1], face[2], face[3])

