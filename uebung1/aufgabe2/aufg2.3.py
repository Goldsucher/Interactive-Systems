# Diese Code beinhaltet Aufgaben 2.3 vom Übungsblatt 1
# Stephan Wagner s853668

import numpy as np
import cv2

img_grey = cv2.imread('resources/images/lenna.jpg', 0)

#https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
# https://handmap.github.io/gradients-and-edge-detection/
blurred_img = cv2.GaussianBlur(img_grey, (5, 5), 0)

sobelx = cv2.Sobel(blurred_img,cv2.CV_64F,1,0)
sobely = cv2.Sobel(blurred_img,cv2.CV_64F,0,1)
sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))


sobelCombined = cv2.bitwise_or(sobelx, sobely)

cv2.imshow("SobelCombined", sobelCombined)
cv2.waitKey(0)
cv2.destroyAllWindows()