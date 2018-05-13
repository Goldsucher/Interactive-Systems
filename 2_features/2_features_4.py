import numpy as np
import cv2
import math
import glob
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
    img_grey = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)

    # compute x and y gradients
    sobelx = cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_grey, cv2.CV_64F, 0, 1, ksize=5)

    # compute magnitude and angle of the gradients
    magnitudes = cv2.magnitude(sobelx, sobely)
    angles = cv2.phase(sobelx, sobely, angleInDegrees=True)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    # count = 0
    for kp in keypoints:
        # print(kp.pt, kp.size)
        (kpx, kpy) = kp.pt

        # extract angle in keypoint sub window
        # angle = angle[int(kpx)][int(kpy)]
        # print(angles)

        # extract gradient magnitude in keypoint subwindow
        # magnitude = magnitude[int(kpx)][int(kpy)]
        # print(magnitudes)

        # create histogram of angle in sub window BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        angle_for_histogram = []
        radius = int(kp.size/2)
        for x in range(int(kpx-radius), int(kpx)):
            for y in range(int(kpy-radius), int(kpy)):
                if ((x-kpx)**2 + (y-kpy)**2) <= (radius*radius):
                    x_sym = int(kpx - (x - kpx))
                    y_sym = int(kpy - (y - kpy))
                    # (x, y), (x, y_sym), (x_sym, y), (x_sym, y_sym) are in the circle
                    # print(x, y, x_sym, y_sym)
                    if magnitudes[x][y] != 0:
                        angle_for_histogram.append(angles[x][y])
                    if magnitudes[x][y_sym] != 0:
                        angle_for_histogram.append(angles[x][y_sym])
                    if magnitudes[x_sym][y] != 0:
                        angle_for_histogram.append(angles[x_sym][y])
                    if magnitudes[x_sym][y_sym] != 0:
                        angle_for_histogram.append(angles[x_sym][y_sym])
        if magnitudes[int(kpx)][int(kpy)] != 0:
            angle_for_histogram.append(angles[int(kpx)][int(kpy)])
            # print("x,y: ", angles[x][y])

        print("collected angles: ", angle_for_histogram)
        (hist, bins) = np.histogram(angle_for_histogram)
        print("histogram: ", hist)
        print("raw bins: ", bins)

        # cv2.imshow('', imgcolor)
        # cv2.waitKey()
        plot_histogram(hist, bins)
        # descr[count] = hist
        # count += 1

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]
# test for all test images
images = glob.glob('./resources/images/hog_test/*.jpg')
for img in images:
    print("for img: ", img)
    test = cv2.imread(img)
    descriptor = compute_simple_hog(test, keypoints)

# cv2.imshow('', result)
# key = cv2.waitKey()
# cv2.destroyAllWindows()
