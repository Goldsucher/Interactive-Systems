# Diese Code beinhaltet Aufgaben 2.4 vom Übungsblatt 1
# Stephan Wagner s853668

import numpy as np
import cv2

#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

def calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        imgCorners = cv2.drawChessboardCorners(frame, (7, 6), corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite('resources/images/imgBeforeCalibration.png', imgCorners)
        cv2.imwrite('resources/images/imgAfterCalibration.png', dst)

        return imgCorners

    if ret == False:
        return img


cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()

mode = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(50)
    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('c'):
        if mode == 1:
            mode = 0
        else:
            mode = 1

    if mode == 1:
        img = calibration()

    cv2.imshow('frame', img)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()