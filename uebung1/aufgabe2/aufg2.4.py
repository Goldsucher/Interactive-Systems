# Diese Code beinhaltet Aufgaben 2.4 vom Übungsblatt 1
# Stephan Wagner s853668

import numpy as np
import cv2

img = cv2.imread('resources/images/lenna.jpg', 1)

# a)

#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#https://stackoverflow.com/questions/21210479/converting-from-rgb-to-lab-colorspace-any-insight-into-the-range-of-lab-val
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# b)
#https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html

#adaptive Gaussion-Treshholding -> gTh
img_grey = cv2.imread('resources/images/lenna.jpg', 0)
img_grey = cv2.medianBlur(img_grey,5)
gTh = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


#adaptive Gaussion-Treshholding -> oTh
ret,oTh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# c)
#https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
canny = cv2.Canny(img_grey,100,200)



cv2.imshow("HSV", hsv)
cv2.imshow("LAB", lab)
cv2.imshow("YUV", yuv)
cv2.imshow("Adaptive Gaussian Thresholding", gTh)
cv2.imshow("Otsu Thresholding", oTh)
cv2.imshow("Canny Edge Detection", canny)

cv2.waitKey()
cv2.destroyAllWindows()