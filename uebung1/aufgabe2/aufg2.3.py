# Diese Code beinhaltet Aufgaben 2.3 vom Übungsblatt 1
# Stephan Wagner s853668

import numpy as np
import cv2

def selfMadeGaussianBlur(img):
    print("do Blur")
    tmpImg = img
    ysizeTmpImg, xsizeTmpImg = tmpImg.shape
    kernel = np.matrix('0.0625 0.125 0.0625; 0.125 0.25 0.125; 0.0625 0.125 0.0625')
    ysizeKernel, xsizeKernel = kernel.shape
    y=1
    x=1
    for row in tmpImg:
        if(y<ysizeTmpImg-1):
            for pixel in row:
                if (x < xsizeTmpImg-1):
                    currentPixel = pixel
                    tmpMatrix = np.matrix([[tmpImg[y-1][x-1]*kernel[0,0], tmpImg[y-1][x]*kernel[0,1], tmpImg[y-1][x+1]*kernel[0,2]],
                                          [tmpImg[y][x-1]*kernel[1,0], currentPixel*kernel[1,1], tmpImg[y][x+1]*kernel[1,2]],
                                         [tmpImg[y+1][x-1]*kernel[2,0], tmpImg[y+1][x]*kernel[2,1], tmpImg[y+1][x+1]*kernel[2,2]]])

                    tmpResultSum = tmpMatrix.sum()
                    tmpImg[y][x] = tmpResultSum
                    x+=1
            x=0
            y+=1
    return tmpImg

def selfMadeSobel(img,x=0, y=0):
    print("do Sobel")
    ysizeImg, xsizeImg = img.shape

    tmpImg = img
    sobelX = np.matrix('1 0 -1;'
                       '2 0 -2;'
                       '1 0 -1')

    sobelY = np.matrix('1 2 1; '
                       '0 0 0;'
                       '-1 -2 -1')


    #kernel = np.matrix('-3 0 3;'
    #                   '-10 0 10;'
    #                   '-3 0 3')

    y = 1
    x = 1
    for row in img:
        if (y < ysizeImg - 1):
            for pixel in row:
                if (x < xsizeImg - 1):
                    currentPixel = pixel
                    #print(currentPixel)
                    tmpMatrixX = np.matrix([[img[y - 1][x - 1] * sobelX[0, 0], img[y - 1][x] * sobelX[0, 1],img[y - 1][x + 1] * sobelX[0, 2]],
                                           [img[y][x - 1] * sobelX[1, 0], currentPixel * sobelX[1, 1],img[y][x + 1] * sobelX[1, 2]],
                                           [img[y + 1][x - 1] * sobelX[2, 0], img[y + 1][x] * sobelX[2, 1],img[y + 1][x + 1] * sobelX[2, 2]]])

                    tmpMatrixY = np.matrix([[img[y - 1][x - 1] * sobelY[0, 0], img[y - 1][x] * sobelY[0, 1], img[y - 1][x + 1] * sobelY[0, 2]],
                                           [img[y][x - 1] * sobelY[1, 0], currentPixel * sobelY[1, 1],img[y][x + 1] * sobelY[1, 2]],
                                           [img[y + 1][x - 1] * sobelY[2, 0], img[y + 1][x] * sobelY[2, 1], img[y + 1][x + 1] * sobelY[2, 2]]])

                    tmpMatrix2 = np.matrix([[img[y - 1][x - 1], img[y - 1][x],
                                             img[y - 1][x + 1] ],
                                           [img[y][x - 1], currentPixel,
                                            img[y][x + 1]],
                                           [img[y + 1][x - 1], img[y + 1][x],
                                            img[y + 1][x + 1]]])

                    tmpResultSumX = np.sum(tmpMatrixX)
                    tmpResultSumY = np.sum(tmpMatrixY)
                    tmpResult = np.ceil(np.sqrt((tmpResultSumX * tmpResultSumX) + (tmpResultSumY * tmpResultSumY)))
                    tmpImg[y][x] = tmpResult
                    x += 1
            x = 0
            y += 1

    return tmpImg




img_grey = cv2.imread('resources/images/lenna.jpg', 0)
cv2.imshow("original", img_grey)
print("show original")
blurred_img = selfMadeGaussianBlur(img_grey)
cv2.imshow("Blur", blurred_img)
print("show blurred")
sobelx = selfMadeSobel(blurred_img,1,0)
cv2.imshow("SobelX", sobelx)
print("show SobelX")
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
sobely = selfMadeSobel(blurred_img,0,1)
cv2.imshow("SobelY", sobely)
print("show SobelY")

#https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
# https://handmap.github.io/gradients-and-edge-detection/
#blurred_img = cv2.GaussianBlur(img_grey, (5, 5), 0)

#sobelx = cv2.Sobel(blurred_img,cv2.CV_64F,1,0)
#sobely = cv2.Sobel(blurred_img,cv2.CV_64F,0,1)
#sobelx = np.uint8(np.absolute(sobelx))
#sobely = np.uint8(np.absolute(sobely))


#orig_blurred = cv2.bitwise_or(sobelx, sobely)
#orig_blurred = cv2.bitwise_or(sobelx, sobely)


#cv2.imshow("SobelCombined", sobelCombined)
