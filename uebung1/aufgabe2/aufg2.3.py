# Diese Code beinhaltet Aufgaben 2.3 vom Übungsblatt 1
# Stephan Wagner s853668

import numpy as np
import cv2


def img2double(img):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype('float') - min_val) / (max_val - min_val)

    return out

def make_gaussian_kernel(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)


def convolution_2d(img, kernel):
    tmpImg = img.copy()
    ysizeTmpImg, xsizeTmpImg = tmpImg.shape
    ysizeKernel, xsizeKernel = kernel.shape
    tmpMatrix = np.zeros((xsizeKernel, ysizeKernel))

    y=1
    x=1
    for row in tmpImg:
        if(y<ysizeTmpImg-1):
            for pixel in row:
                if (x < xsizeTmpImg-1):

                    i = int(np.trunc(xsizeKernel/2))*(-1)
                    for xTmpMatrix in range(0, int(np.trunc(xsizeKernel/2))+2):
                        j = int(np.trunc(ysizeKernel / 2)) * (-1)
                        for yTmpMatrix in range(0, int(np.trunc(ysizeKernel/2))+2):
                            tmpMatrix[xTmpMatrix][yTmpMatrix] = img[y+i][x+j] * kernel[xTmpMatrix, yTmpMatrix]
                            j+=1
                        i+=1

                    tmpResultSum = tmpMatrix.sum()
                    tmpImg[y][x] = tmpResultSum
                    x+=1
            x=0
            y+=1
    return tmpImg

def main():
    print("Operations are performed, please wait...")
    img_grey_orig = cv2.imread('resources/images/lenna.jpg', 0)
    img_grey = img_grey_orig.copy()
    print("do Blur 3x3")
    img_blurred_3 = convolution_2d(img2double(img_grey), make_gaussian_kernel(3))
    print("do Blur 5x5")
    img_blurred_5 = convolution_2d(img2double(img_grey), make_gaussian_kernel(5))

    sobelX_kernel = np.matrix('1 0 -1;2 0 -2; 1 0 -1')
    sobelY_kernel = np.matrix('1 2 1;0 0 0;-1 -2 -1')

    print("do Sobel X")
    img_sobelX=convolution_2d(img2double(img_grey), sobelX_kernel)

    print("do Sobel Y")
    img_sobelY=convolution_2d(img2double(img_grey), sobelY_kernel)

    print("combine SobelX + SobelY")
    img_magnitude_gradients = np.sqrt(np.power(img_sobelX, 2) + np.power(img_sobelY, 2))

    img_orig_blurred = np.concatenate((img2double(img_grey_orig), img_blurred_3, img_blurred_5), axis=1)
    img_sobel = np.concatenate((img_sobelX, img_sobelY, img_magnitude_gradients), axis=1)
    img_all= np.concatenate((img_orig_blurred, img_sobel), axis=0)

    cv2.imshow("Images", img_all)
    print("Show all Operations")
    print("DONE!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
