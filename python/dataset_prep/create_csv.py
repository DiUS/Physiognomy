#!/usr/bin/env python

import sys
import os.path
import csv

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "usage: create_csv <base_path to top of Emotion directory tree>"
        sys.exit(1)

    BASE_PATH=sys.argv[1]
    SEPARATOR=","

    emotion_lookup = {0: 'neutral', 1 : 'anger', 2 : 'contempt', 3 : 'disgust', 4 : 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise' }

    with open('classified_images.csv','wb') as csvfile:
      file_writer = csv.writer(csvfile, delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
      for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                if '.txt' in filename:
                  emotion_path = os.path.join(dirname, subdirname,filename)
                  fh = open(emotion_path,"r")
                  emotion_int = int(float(fh.readline().rstrip()))
                  png_filename = filename.replace('_emotion.txt','.png')
                  png_path = subject_path.replace('Emotion','cohn-kanade-images')
                  abs_path = "%s/%s" % (png_path, png_filename)
                  file_writer.writerow([abs_path, emotion_lookup[emotion_int]])
                  