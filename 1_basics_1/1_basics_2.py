import numpy as np
import cv2


def translation(img, value=10):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1,0,value], [0,1,0]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    return img_translation;
    #cv2.imshow('Translation', img_translation)

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
    img2_orig = cv2.imread('resources/images/lenna.jpg', 1)  # 1 = farbe, 0 = grau (mittelwert von rgb) und -1 = with alphachannel
    img_grey = cv2.imread('resources/images/lenna.jpg', 0)
    #print(img.shape[:2])  # Anzahl Reihe und Spalte

    img_grey = prepareGreyImage(img2_orig, img_grey)    # aus 1 grauwert werden 3, damit es mit rgb compatible ist und merge funkt
    bothImg = mergeImages(img_grey, img_orig)

    #cv2.imwrite('resources/images/lenna_both.png',vis) # save image

    translationValue = 0;
    rotationValue = 0;
    cv2.imshow('Image', bothImg)  # ausgabe
    while (1):
        key = cv2.waitKey(0)
        print(key)
        if key == 113:       #ascii value for q
            print('q')
            cv2.destroyAllWindows()
            break
        elif key == 116:         #ascii value for t
            translationValue +=10;
            bothImg = translation(bothImg, translationValue)
            cv2.imshow('Image', bothImg)
        elif key == 114:    #ascii value for r
            rotationValue += 10;
            bothImg = rotation(bothImg, rotationValue)
            cv2.imshow('Image', bothImg)

main()