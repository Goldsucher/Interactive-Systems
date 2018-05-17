# Diese Code beinhaltet Aufgabe 2 vom Übungsblatt 2
# Stephan Wagner s853668

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

imageStitcher = ImageStitcher(imgList)
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # output all matching images
    # output result
    # Note: if necessary resize the image
    for img in matchlist + result:
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('', img)
        cv2.waitKey()

    cv2.destroyAllWindows()


