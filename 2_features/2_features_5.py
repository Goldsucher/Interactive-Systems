import numpy as np
import cv2
import sys

###############################################################
#
# Eigenvector
#
###############################################################

# Given three very simple images

# 3x3 edge image
edge = np.zeros((3, 3, 1), np.float32)
edge[1][0] = 255.0
edge[1][1] = 255.0
edge[1][2] = 255.0
edge[2][0] = 255.0
edge[2][1] = 255.0
edge[2][2] = 255.0

# 3x3 corner image
corner = np.zeros((3, 3, 1), np.float32)
corner[0][0] = 0.0
corner[0][1] = 0.0
corner[0][2] = 0.0
corner[1][0] = 1.0
corner[1][1] = 0.95
corner[1][2] = 0.0
corner[2][0] = 1.0
corner[2][1] = 0.95
corner[2][2] = 0.0

# 3x3 flat region
flat = np.zeros((3, 3, 1), np.float32)

# choose which one to use to compute eigenvector / eigenvalues
# img = edge
for key, img in {'edge': edge, 'corner': corner, 'flat': flat}.items():

    # simple gradient extraction
    k = np.matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ktrans = k.transpose()

    Gx = cv2.filter2D(img, -1, k)
    Gy = cv2.filter2D(img, -1, ktrans)

    # print("dx / dy", Gx, Gy)

    # this is the 2x2 matrix we need to evaluate
    # the Harris corners
    eigMat = np.zeros((2, 2), np.float32)

    # compute values for matrix eigMat and fill matrix with
    # necessary values

    # YOUR CODE HERE
    eigMat[0][0] = np.sum(Gx ** 2)
    eigMat[0][1] = np.sum(Gx * Gy)
    eigMat[1][0] = np.sum(Gy * Gx)
    eigMat[1][1] = np.sum(Gy ** 2)
    # print(eigMat)

    # compute eigenvectors and eigenvalues using the numpy
    # linear algebra package

    # YOUR CODE HERE
    w, v = np.linalg.eigvals(eigMat)

    # out and show the image
    # print("matrix:", eigMat, '\n')
    print("img: ", key)
    print("eigvalues: ", w)
    print("eigenvecv: ", v)
    print()
    # scaling_factor = 100
    # img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)