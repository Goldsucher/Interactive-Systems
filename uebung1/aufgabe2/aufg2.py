import numpy as np
import cv2

img = cv2.imread('resources/images/lenna.jpg', 1)  # 1 = farbe, 0 = grau (mittelwert von rgb) und -1 = with alphachannel
img2 = cv2.imread('resources/images/lenna.jpg', 1)  # 1 = farbe, 0 = grau (mittelwert von rgb) und -1 = with alphachannel
img_grey = cv2.imread('resources/images/lenna.jpg', 0)
#print(img.shape[:2])  # Anzahl Reihe und Spalte

i=0
for row in img_grey:
    j=0
    for col in row:
        img2[i][j][0] = col
        img2[i][j][1] = col
        img2[i][j][2] = col
        j+=1
    i+=1


bothImg = np.concatenate((img2, img), axis=1) # 1=vertikal 0 = horizontal
#cv2.imwrite('resources/images/lenna_both.png',vis)

def translation(img, value=10):
    num_rows, num_cols = bothImg.shape[:2]
    translation_matrix = np.float32([[1,0,value], [0,1,0]])
    img_translation = cv2.warpAffine(bothImg, translation_matrix, (num_cols, num_rows))
    cv2.imshow('Translation', img_translation)




#cv2.waitKey()
cv2.imshow('bothImg', bothImg)
cv2.waitKey(0)
cv2.destroyAllWindows()