import numpy as np
import cv2
import math
import sys
from ImageStitcher import *


############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images

# order of input images is important is important (from right to left)
imageStitcher = ImageStitcher(["resources/images/pano3.jpg",
                               "resources/images/pano2.jpg",
                               "resources/images/pano1.jpg",
                               "resources/images/pano6.jpg",
                               "resources/images/pano5.jpg",
                               "resources/images/pano4.jpg"])

(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image
    cv2.namedWindow('Aufgabe 2.2')
    for img in matchlist+result:
        # crop black around img (https://goo.gl/cWwic9)
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = bw.shape
        padding = 0
        non_empty_columns = np.where(bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(bw.max(axis=1) > 0)[0]
        cropBox = (min(non_empty_rows) * (1 - padding),
                   min(max(non_empty_rows) * (1 + padding), rows),
                   min(non_empty_columns) * (1 - padding),
                   min(max(non_empty_columns) * (1 + padding), cols))
        img = img[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]

        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('', img)
        cv2.waitKey()

    cv2.destroyAllWindows()


