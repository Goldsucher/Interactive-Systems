import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints):

    # convert color to gray image and extract feature in gray
    imggray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(imggray, None)

    # compute x and y gradients
    sobelx = cv2.Sobel(imggray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(imggray, cv2.CV_64F, 0, 1, ksize=5)

    # compute magnitude and angle of the gradients
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        print(kp.pt, kp.size)
        # extract angle in keypoint sub window

        # extract gradient magnitude in keypoint subwindow


        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        (hist, bins) = np.histogram(imggray(), 256, [0, 256])

        plot_histogram(hist, bins)

        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
test = cv2.imread('./images/hog_test/diag.jpg')
descriptor = compute_simple_hog(test, keypoints)



