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
pano1 = cv2.imread("resources/images/pano1.jpg", 1)
pano2 = cv2.imread("resources/images/pano2.jpg", 1)
pano3 = cv2.imread("resources/images/pano3.jpg", 1)
pano4 = cv2.imread("resources/images/pano4.jpg", 1)
pano5 = cv2.imread("resources/images/pano5.jpg", 1)
pano6 = cv2.imread("resources/images/pano6.jpg", 1)

# order of input images is important is important (from right to left)
imageStitcher = ImageStitcher([pano6, pano5]) # list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # YOUR CODE HERE
    cv2.imshow("Test", imageStitcher)
    # output all matching images
    # output result
    # Note: if necessary resize the image

