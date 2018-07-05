# Blatt 4 "4_ml"
# Aufgabe 1: Bildklassifikation / Support Vector Machine / Deep Learning
# Steffen Burlefinger (859077)

import numpy as np
import cv2
import glob
from sklearn import svm


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def create_keypoints(w, h):

    keypoints = []
    keypoint_size = 15   # has to be odd

    radius = int(keypoint_size/2)
    step = radius*2

    print("Creating keypoints: kp-size = " + str(keypoint_size) + "px; kp-numbers = " + str(w) + "x" + str(h))
    for cx in range(radius, w, step):
        for cy in range(radius, h, step):
            keypoints.append(cv2.KeyPoint(cx, cy, keypoint_size))

    return keypoints

# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px

sift = cv2.xfeatures2d.SIFT_create()
key_points = create_keypoints(256, 256)

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers

labels = {1: 'flowers', 2: 'cars', 3: 'faces'}
train = []

for label_int, label_annotations in labels.items():
    imgs_paths = glob.glob('images/db/train/'+label_annotations+'/*.jpg')
    for img_path in imgs_paths:
        img = cv2.imread(img_path, 1)
        kp, descr = sift.compute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), key_points)
        train.append((label_int, descr.ravel(), img))

y_train = train[0][0]
x_train = np.asmatrix(train[0][1])
for y, X, _ in train[1:]:
    x_train = np.vstack((x_train, X))
    y_train = np.vstack((y_train, y))

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)

print('\nLearning phase initiated ... ')
clf = svm.LinearSVC()
clf.fit(x_train, y_train.ravel())
print('Learning phase finished ...')


# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image

test_images = glob.glob('images/db/test/*.jpg')

print("\nTesting phase:")
for test_img_path in test_images:
    test_img = cv2.imread(test_img_path, 1)
    test_kp, test_dsc = sift.compute(test_img, key_points)
    test_dsc = test_dsc.ravel()
    prediction = clf.predict([test_dsc])

    # 5. output the class + corresponding name
    print("The Image " + test_img_path + " belongs to class: " + str(prediction[0]) + " = " + labels.get(prediction[0]))
