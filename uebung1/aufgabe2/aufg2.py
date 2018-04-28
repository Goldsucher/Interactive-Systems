# Diese Code beinhaltet Aufgaben 2.1 und 2.2 vom Übungsblatt 1
# Stephan Wagner s853668
import numpy as np
import cv2

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
def translation(img, value=10):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1,0,value], [0,1,0]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    return img_translation;

# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
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

    return np.concatenate((img, img2), axis=1) # 1=vertikal 0 = horizontal

# -- MainProgramm ---
def main():
    img_orig = cv2.imread('resources/images/lenna.jpg', 1)  # 1 = farbe, 0 = grau (mittelwert von rgb) und -1 = with alphachannel
    img2_orig = cv2.imread('resources/images/lenna.jpg', 1)
    img_grey = cv2.imread('resources/images/lenna.jpg', 0)
    #print(img.shape[:2])  # Anzahl Reihe und Spalte

    img_grey = prepareGreyImage(img2_orig, img_grey)
    bothImg = mergeImages(img_grey, img_orig)

    #cv2.imwrite('resources/images/lenna_both.jpg',vis) # save image

    translationValue = 0;
    rotationValue = 0;
    cv2.imshow('Image', bothImg)
    while (1):
        bothImg = mergeImages(img_grey, img_orig)
        key = cv2.waitKey(0)
        # https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv
        if key == 113:       #ascii value for q
            cv2.destroyAllWindows()
            break
        elif key == 116:         #ascii value for t
            translationValue +=10;
            bothImg = translation(bothImg, translationValue)
            bothImg = rotation(bothImg, rotationValue)
            cv2.imshow('Image', bothImg)
        elif key == 114:    #ascii value for t
            rotationValue += 10;
            bothImg = translation(bothImg, translationValue)
            bothImg = rotation(bothImg, rotationValue)
            cv2.imshow('Image', bothImg)

main()