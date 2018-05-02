# Diese Code beinhaltet Aufgaben 2.4 vom Übungsblatt 1
# Stephan Wagner s853668

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1
    if ch == ord('2'):
        mode = 2
    if ch == ord('3'):
        mode = 3
    if ch == ord('4'):
        mode = 4
    if ch == ord('5'):
        mode = 5
    if ch == ord('6'):
        mode = 6
    if ch == ord('q'):
        break

    if mode == 1:
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mode == 2:
        # https://stackoverflow.com/questions/21210479/converting-from-rgb-to-lab-colorspace-any-insight-into-the-range-of-lab-val
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    if mode == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    if mode == 4:
        # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
        # adaptive Gaussion-Treshholding
         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
         frame = cv2.medianBlur(frame,5)
         frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    if mode == 5:
        # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
        # adaptive Gaussion-Treshholding
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if mode == 6:
        # Canny-Edge
        # https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.Canny(frame,100,200)

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



# import numpy as np
# import cv2
#
# img = cv2.imread('resources/images/lenna.jpg', 1)
#
# # a)
#
# #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# #https://stackoverflow.com/questions/21210479/converting-from-rgb-to-lab-colorspace-any-insight-into-the-range-of-lab-val
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#
# yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#
# # b)
# #https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
#
# #adaptive Gaussion-Treshholding -> gTh
# img_grey = cv2.imread('resources/images/lenna.jpg', 0)
# img_grey = cv2.medianBlur(img_grey,5)
# gTh = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#
#
# #adaptive Gaussion-Treshholding -> oTh
# ret,oTh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# # c)
# #https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
# canny = cv2.Canny(img_grey,100,200)
#
#
#
# cv2.imshow("HSV", hsv)
# cv2.imshow("LAB", lab)
# cv2.imshow("YUV", yuv)
# cv2.imshow("Adaptive Gaussian Thresholding", gTh)
# cv2.imshow("Otsu Thresholding", oTh)
# cv2.imshow("Canny Edge Detection", canny)
#
# cv2.waitKey()
# cv2.destroyAllWindows()