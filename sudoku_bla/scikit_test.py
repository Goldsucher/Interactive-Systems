import numpy as np
import cv2
import mnist

from PIL import Image
from sklearn import svm
from numpy import array


def create_keypoints(w, h):
    keypoints = []
    keypoint_size = 3  # has to be odd

    radius = int(keypoint_size / 2)
    step = radius * 2

    print("Creating keypoints: kp-size = " + str(keypoint_size) + "px; kp-numbers = " + str(w) + "x" + str(h))
    for cx in range(radius, w, step):
        for cy in range(radius, h, step):
            keypoints.append(cv2.KeyPoint(cx, cy, keypoint_size))

    return keypoints


def transform(array, h, w):
    x = 0
    y = 0
    result = []
    for image in array:
        for pix in image:
            if y < w:
                data[x, y] = [pix, pix, pix]
                y += 1
            if y >= h:
                x += 1
                y = 0
        tmp = np.append(result, Image.fromarray(data, 'RGB'))
        #tmp = np.array(data)
        result = np.append(tmp, data)
        x = 0
        y = 0

    return result


sift = cv2.xfeatures2d.SIFT_create()
key_points = create_keypoints(28, 28)

x_train, t_train, x_test, t_test = mnist.load()

wi, he = 28, 28
data = np.zeros((he, wi, 3), dtype=np.uint8)

print("Preprocessing.....")
# print("Transform: X_train")
# new_x_train = transform(x_train, he, wi)
print("Transform: X_test")
#new_x_test = transform(x_test, he, wi)
#print(new_x_test)
# img = new_x_test[0]
# img.show()
# cv2.waitKey()
# cv2.destroyAllWindows()

train = []
indexLabels = 0

# print("train phase ...")
# for x in new_x_test:
#     # img = cv2.imread(x, 1)
#     img = x
#     kp, descr = sift.compute(img, key_points)
#     print(descr)
#     exit()
#     train.append((t_train[indexLabels], descr.ravel(), img))
#     indexLabels += 1
#
# print(train)
# exit()
#
# y_train = train[0][0]
# x_train = np.asmatrix(train[0][1])
# for y, X, _ in train[1:]:
#     x_train = np.vstack((x_train, X))
#     y_train = np.vstack((y_train, y))
#
# print('\nLearning phase initiated ... ')
# clf = svm.LinearSVC()
# clf.fit(x_train, y_train.ravel())
# print('Learning phase finished ...')
#
# print("\nTesting phase:")
# indexLabels = 0

# for test in new_x_test:
#     test_img = cv2.imread(test, 1)
#     test_kp, test_dsc = sift.compute(test_img, key_points)
#     test_dsc = test_dsc.ravel()
#     prediction = clf.predict([test_dsc])
#
#     # 5. output the class + corresponding name
#     print(str(prediction[0]) + " = " + t_test[indexLabels])
#     indexLabels += 1
