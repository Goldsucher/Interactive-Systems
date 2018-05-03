# Blatt 1 "1_basics"
# Aufgabe 2: Computer Vision Basics + OpenCV
# 2.3 - Umsetzung mit OpenCV
# Steffen Burlefinger (859077)

import numpy as np
import cv2

img_grey = cv2.imread("Lenna.png", 0)


# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
img_blurred = cv2.blur(img_grey, (5, 5))


# https://stackoverflow.com/questions/33679738/measure-edge-strength-in-opencv-magnitude-of-gradient
def getGradientMagnitude(img_blurred):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_64F
    sobelx = cv2.Sobel(img_blurred, ddepth, 1, 0)  # x
    sobely = cv2.Sobel(img_blurred, ddepth, 0, 1)  # y
    sobelxabs = cv2.convertScaleAbs(sobelx)
    sobelyabs = cv2.convertScaleAbs(sobely)
    # mag = cv2.addWeighted(sobelxabs, 0.5, sobelyabs, 0.5, 0)
    sobelCombined = cv2.bitwise_or(sobelxabs, sobelyabs)
    return sobelCombined


result = getGradientMagnitude(img_blurred)

while True:
    all = np.concatenate((img_grey, img_blurred, result), axis=1)
    cv2.imshow("Aufgabe 2.3 mit CV", all)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break