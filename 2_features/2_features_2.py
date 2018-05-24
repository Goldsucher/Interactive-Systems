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
imageStitcher = ImageStitcher(["images/pano3.jpg",
                               "images/pano2.jpg",
                               "images/pano1.jpg",
                               "images/pano6.jpg",
                               "images/pano5.jpg",
                               "images/pano4.jpg"])

(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image
    for img in matchlist+result:
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Aufgabe 2.2', img)
        cv2.waitKey()

    cv2.destroyAllWindows()