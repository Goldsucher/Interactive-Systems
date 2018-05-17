import numpy as np
import cv2
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
    # angles = cv2.phase(sobelx, sobely, angleInDegrees=True)
    angles = cv2.phase(sobelx, sobely)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        # create histogram of angle in sub window BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        sub_window_magnitudes = magnitudes[10:20:, 10:20:]
        sub_window_angles = angles[10:20:, 10:20:]
        angle_for_histogram = np.extract(sub_window_magnitudes, sub_window_angles)

        # print("collected angles: ", angle_for_histogram)
        (hist, bins) = np.histogram(angle_for_histogram, np.linspace(0, 2 * np.pi, 9))
        # print("histogram: ", hist)

        plot_histogram(hist, bins)
        descr[count] = hist
        count += 1

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]
# test for all test images
images = glob.glob('./resources/images/hog_test/*.jpg')
for img in images:
    print("for img: ", img)
    test = cv2.imread(img)
    descriptor = compute_simple_hog(test, keypoints)
    print(descriptor)

# cv2.imshow('', result)
# key = cv2.waitKey()
# cv2.destroyAllWindows()
