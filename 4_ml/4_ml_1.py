import numpy as np
import cv2
import glob, sys
from sklearn import svm

############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################


def create_key_points(w, h):
    key_points_created = []
    key_point_size = 15   # has to be odd cause we need a mid point

    radius = int(key_point_size/2)
    step = radius * 2

    # creating key point grid
    for cx in range(radius, w, step):
        for cy in range(radius, h, step):
            key_points_created.append(cv2.KeyPoint(cx, cy, key_point_size))

    return key_points_created


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 key points on each image with sub window of 15x15px
print("\nCreating key points ... ", end="")
sys.stdout.flush()
sift = cv2.xfeatures2d.SIFT_create()
key_points = create_key_points(256, 256)
print("done.")
sys.stdout.flush()

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
labels = {1: 'flowers', 2: 'cars', 3: 'faces'}
train = []

print("Create descriptors and flatter ... ", end="")
sys.stdout.flush()
for label_int, label_annotations in labels.items():
    train_images_with_full_path = glob.glob('resources/images/db/train/' + label_annotations + '/*.jpg')
    for train_image_with_full_path in train_images_with_full_path:
        train_image = cv2.imread(train_image_with_full_path, 1)
        kp, descr = sift.compute(cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY), key_points)
        train.append((label_int, descr.ravel(), train_image))

y_train = train[0][0]
x_train = np.asmatrix(train[0][1])
for y, X, _ in train[1:]:
    x_train = np.vstack((x_train, X))
    y_train = np.vstack((y_train, y))
print('done.')
sys.stdout.flush()

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the
#  documentation. You will need .fit(X_train, y_train)
print("Learning ... ", end="")
sys.stdout.flush()
clf = svm.LinearSVC()
clf.fit(x_train, y_train.ravel())
print('done.')
sys.stdout.flush()


# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
test_images_with_full_path = glob.glob('resources/images/db/test/*.jpg')

print("\nTesting ... ")
for test_image_with_full_path in test_images_with_full_path:
    test_image = cv2.imread(test_image_with_full_path, 1)
    test_kp, test_dsc = sift.compute(test_image, key_points)
    test_dsc = test_dsc.ravel()
    prediction = clf.predict([test_dsc])

    # 5. output the class + corresponding name
    print("Image " + test_image_with_full_path + " is classified to " + labels.get(prediction[0]))
