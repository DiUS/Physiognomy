FaceRec takes two ';' seperated files, where the first column is the filename and the second is an integer representing the label (class)

The first file is a list of training data, the second is a list of test data.

This will then create a 'predictions' folder and name the predicted classes for the test data.

Currently this assumes that a label of 0 = Happy and 1 = Sad