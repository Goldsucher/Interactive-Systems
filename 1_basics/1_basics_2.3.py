# Blatt 1 "1_basics"
# Aufgabe 2: Computer Vision Basics + OpenCV
# 2.3
# Steffen Burlefinger (859077)

import numpy as np
import cv2


def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.
    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)

    return out


def make_gaussian(size, fwhm = 3, center = None):
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

    y = 1
    x = 1
    for row in tmpImg:
        if(y < ysizeTmpImg - 1):
            for pixel in row:
                if (x < xsizeTmpImg - 1):

                    i = int(np.trunc(xsizeKernel / 2)) * (-1)
                    for xTmpMatrix in range(0, int(np.trunc(xsizeKernel / 2)) + 2):
                        j = int(np.trunc(ysizeKernel / 2)) * (-1)
                        for yTmpMatrix in range(0, int(np.trunc(ysizeKernel / 2)) + 2):
                            tmpMatrix[xTmpMatrix][yTmpMatrix] = img[y + i][x + j] * kernel[xTmpMatrix, yTmpMatrix]
                            j += 1
                        i += 1

                    tmpResultSum = tmpMatrix.sum()
                    tmpImg[y][x] = tmpResultSum
                    x+=1
            x = 0
            y += 1
    return tmpImg


# 15 x 15
# img_grey = cv2.imread("Lenna_klein.png", 0)

# 256 x 256
img_grey = cv2.imread("Lenna_medium.png", 0)

# 512 x 512
# img_grey = cv2.imread("Lenna.png", 0)


# print("3x3 blur")
# img_blurred_3 = convolution_2d(im2double(img_grey),make_gaussian(3))
print("5x5 blur")
img_blurred_5 = convolution_2d(im2double(img_grey), make_gaussian(5))

sobelX_kernel = np.matrix('1 0 -1; 2 0 -2; 1 0 -1')
sobelY_kernel = np.matrix('1 2 1;0 0 0;-1 -2 -1')

print("Sobel X")
img_sobelX = convolution_2d(im2double(img_grey), sobelX_kernel)

print("Sobel Y")
img_sobelY = convolution_2d(im2double(img_grey), sobelY_kernel)

print("Sobel X + Sobel Y")
img_magnitude_gradients = np.sqrt(np.power(img_sobelX, 2) + np.power(img_sobelY, 2))

# prepare for concatenation
img_grey = im2double(img_grey)

# concatenation
img_orig_blurred = np.concatenate((img_grey, img_grey, img_blurred_5), axis=1)
img_sobel = np.concatenate((img_sobelX, img_sobelY, img_magnitude_gradients), axis=1)
result = np.concatenate((img_orig_blurred, img_sobel), axis=0)

# output
cv2.imshow("Aufgabe 2.3", result)
print("show  result")
cv2.waitKey(0)
cv2.destroyAllWindows()

