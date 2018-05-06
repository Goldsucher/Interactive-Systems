# Blatt 1 "1_basics"
# Aufgabe 2: Computer Vision Basics + OpenCV
# 2.1 + 2.2
# Steffen Burlefinger (859077)

import numpy as np
import cv2


def translation(img, value=10):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1,0,value], [0,1,0]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    return img_translation


def rotation(img, angle=10):
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  img_rotation = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

  return img_rotation


def prepareGreyImage(img, img_grey):
    i = 0
    for row in img_grey:
        j = 0
        for col in row:
            img[i][j][0] = col
            img[i][j][1] = col
            img[i][j][2] = col
            j += 1
        i += 1

    return img


def mergeImages(img, img2):

    return np.concatenate((img, img2), axis=1)



def main():
    img_orig = cv2.imread('Lenna.png', 1)  # 0 = Graustufen, 1 = Farbe, -1 = mit Alphachannel
    img2_orig = cv2.imread('Lenna.png', 1)
    img_grey = cv2.imread('Lenna.png', 0)
    # print(img.shape[:2])  # Anzahl Reihe und Spalte

    img_grey = prepareGreyImage(img2_orig, img_grey)
    bothImg = mergeImages(img_grey, img_orig)


    translationValue = 0
    rotationValue = 0
    cv2.imshow('Aufgabe 2.1_2.2', bothImg)
    while True:
        key = cv2.waitKey()
        print(key)
        bothImg = mergeImages(img_grey, img_orig)
        if key == 113:  # ascii
            print('q')
            cv2.destroyAllWindows()
            break
        elif key == 116:
            translationValue += 10
            bothImg = translation(bothImg, translationValue)
            bothImg = rotation(bothImg, rotationValue)
            cv2.imshow('Aufgabe 2.1_2.2', bothImg)
        elif key == 114:
            rotationValue += 10
            bothImg = translation(bothImg, translationValue)
            bothImg = rotation(bothImg, rotationValue)
            cv2.imshow('Aufgabe 2.1_2.2', bothImg)

main()