# Diese Code beinhaltet Aufgaben 2 vom Übungsblatt 2
# Stephan Wagner s853668

import numpy as np
import cv2
import math
import sys
from ImageStitcher import *

# https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images
# order of input images is important is important (from right to left)
imgDir = 'resources/images/pano/'
imgList = [imgDir+'pano3.jpg',
          imgDir+'pano2.jpg',
          imgDir+'pano1.jpg',
          imgDir+'pano6.jpg',
          imgDir+'pano5.jpg',
          imgDir+'pano4.jpg']

imageStitcher = ImageStitcher(imgList)# list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image
    for img in matchlist + result:
        # crop black around img (https://goo.gl/cWwic9)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # rows, cols = img_gray.shape
        # padding = 0
        # non_empty_columns = np.where(img_gray.max(axis=0) > 0)[0]
        # non_empty_rows = np.where(img_gray.max(axis=1) > 0)[0]
        # cropBox = (min(non_empty_rows) * (1 - padding),
        #            min(max(non_empty_rows) * (1 + padding), rows),
        #            min(non_empty_columns) * (1 - padding),
        #            min(max(non_empty_columns) * (1 + padding), cols))
        # img = img[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]

        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('', img)
        cv2.waitKey()

    cv2.destroyAllWindows()


