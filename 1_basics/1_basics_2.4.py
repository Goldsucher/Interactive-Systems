# Blatt 1 "1_basics"
# Aufgabe 2: Computer Vision Basics + OpenCV
# 2.4
# Steffen Burlefinger (859077)

import cv2

cap = cv2.VideoCapture(0)
mode = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # mode switch
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
        # adaptive Gaussian-Threshold
         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
         frame = cv2.medianBlur(frame,5)
         frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    if mode == 5:
        # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
        # adaptive Gaussian-Threshold
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if mode == 6:
        # Canny Edge Detection
        # https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.Canny(frame, 100, 300)

    # Display the resulting frame
    cv2.imshow('Aufgabe 2.4', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()