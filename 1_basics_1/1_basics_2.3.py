import numpy as np
import cv2


def apply_kernel(image, kernel):
    kernel_rows, kernel_cols = kernel.shape
    # normalize kernel
    kernel = kernel / np.full(kernel.shape, kernel.sum())
    print("applying "+str(kernel.shape)+" kernel: 0%", end='', flush=True)
    blured = image.copy()

    weight, height = blured.shape
    for i in range(weight*height):
        image_row = int(i / weight)
        image_col = i % weight

        blured[image_col][image_row] = 0

        for kernel_col in range(kernel_cols):
            for kernel_row in range(kernel_rows):
                orig_access_col = image_col + (kernel_col + (int(kernel_cols/2)*-1))
                orig_access_row = image_row + (kernel_row + (int(kernel_rows/2)*-1))
                if -1 < orig_access_row < height and -1 < orig_access_col < weight:
                    blured[image_col][image_row] += (kernel.item(kernel_col, kernel_row) * image[orig_access_col][orig_access_row])

        if i % (weight*height/10) == 0:
            p = int((i / (weight*height))*10+1)
            print(".."+str(p)+"0%", end='', flush=True)

    print("..done.", end='\n', flush=True)
    return blured.copy()


def apply_kernel_raw(image, kernel):
    kernel_rows, kernel_cols = kernel.shape
    print("applying "+str(kernel.shape)+" kernel: 0%", end='', flush=True)
    result_matrix = np.zeros_like(image).astype(int)

    weight, height = image.shape
    for i in range(weight*height):
        image_row = int(i / weight)
        image_col = i % weight

        for kernel_col in range(kernel_cols):
            for kernel_row in range(kernel_rows):
                orig_access_col = image_col + (kernel_col + (int(kernel_cols/2)*-1))
                orig_access_row = image_row + (kernel_row + (int(kernel_rows/2)*-1))
                if -1 < orig_access_row < height and -1 < orig_access_col < weight:
                    result_matrix[image_col][image_row] += (kernel.item(kernel_col, kernel_row) * image[orig_access_col][orig_access_row])

        if i % (weight*height/10) == 0:
            p = int((i / (weight*height))*10+1)
            print(".."+str(p)+"0%", end='', flush=True)

    print("..done.", end='\n', flush=True)
    return result_matrix


def to_image(matrix):
    image = np.zeros_like(matrix).astype(np.uint8)
    weight, height = matrix.shape
    for i in range(weight * height):
        image_row = int(i / weight)
        image_col = i % weight
        new_pixel_value = matrix[image_col][image_row]
        new_pixel_value = 0 if new_pixel_value < 0 else new_pixel_value
        new_pixel_value = 255 if new_pixel_value > 255 else new_pixel_value
        image[image_col][image_row] = new_pixel_value

    return image


def main():
    # original picture
    image_gary = cv2.imread('resources/images/lenna.jpg', 0)
    cv2.putText(image_gary, 'original', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)

    # blur gauss
    kernel_gauss_blur = np.matrix('1 2 1; 2 4 2; 1 2 1')
    image_blur_gauss = apply_kernel(cv2.imread('resources/images/lenna.jpg', 0), kernel_gauss_blur)
    ibg = image_blur_gauss.copy()
    cv2.putText(image_blur_gauss, 'blur gauss', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)

    # blur mean
    kernel_test = np.matrix('1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1')
    kernel_mean_blur = np.matrix('1 1 1; 1 1 1; 1 1 1')
    image_blur_mean = apply_kernel(cv2.imread('resources/images/lenna.jpg', 0), kernel_mean_blur)
    cv2.putText(image_blur_mean, 'blur mean 5x5', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)

    # sobel x
    kernel_sobel_x = np.matrix('1 0 -1; 2 0 -2; 1 0 -1')
    sobel_x = apply_kernel_raw(ibg, kernel_sobel_x)
    image_sobel_x = to_image(sobel_x)
    cv2.putText(image_sobel_x, 'sobel x', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)

    # sobel y
    kernel_sobel_y = np.matrix('1 2 1; 0 0 0; -1 -2 -1')
    sobel_y = apply_kernel_raw(ibg, kernel_sobel_y)
    image_sobel_y = to_image(sobel_y)
    cv2.putText(image_sobel_y, 'sobel y', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)

    # magnitude gradients
    magnitude_gradients = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    image_magnitude_gradients = to_image(magnitude_gradients)
    cv2.putText(image_magnitude_gradients, 'edge detection', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)

    while True:
        image_line1 = np.concatenate((image_gary, image_blur_gauss, image_blur_mean), axis=1)
        image_line2 = np.concatenate((image_sobel_x, image_sobel_y, image_magnitude_gradients), axis=1)
        image_show = np.concatenate((image_line1, image_line2), axis=0)
        cv2.imshow('Abgabe 2.3', image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break


main()
