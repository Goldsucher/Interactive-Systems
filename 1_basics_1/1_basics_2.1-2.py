import numpy as np
import cv2
import sys


def matrix_translation(img, value=10):
    """code taken from goo.gl/nBbvfh (opencv.org)"""
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, value], [0, 1, 0]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return img_translation


def matrix_rotation(img, angle=10):
    """code taken from https://goo.gl/w1cXhS (stackoverflow.com)"""
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img_rotation = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return img_rotation


def main():
    # 3 channel color
    image_color = cv2.imread('resources/images/lenna.jpg', cv2.IMREAD_COLOR)
    # print(image_color.shape[:2])
    # print(image_color[0][0])  # fist pixel

    # 1 channel gray
    image_gray = cv2.imread('resources/images/lenna.jpg', 0);
    # print(image_gray.shape[:2])
    # print(image_gray[0][0])  # fist pixel

    # convert 1 channel gray to 3 channel gray
    image_3ch_gray = np.stack((image_gray,) * 3, -1)
    # print(image_3ch_gray.shape[:2])
    # print(image_3ch_gray[0][0])  # fist pixel

    # merge the color and gray
    merged_image = np.concatenate((image_color, image_3ch_gray), axis=1)

    translation = 0
    rotation = 0

    while True:
        image_show = matrix_translation(merged_image, translation)
        image_show = matrix_rotation(image_show, rotation)
        cv2.imshow('Image', image_show)

        key = cv2.waitKey(0)

        # quit on 'q' or 'Q' or 'ESC'
        if key == 113 or key == 81 or key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)

        # translate on 't' or 'T'
        elif key == 116 or key == 84:
            translation += 10

        # rotate on 'r' or 'R'
        elif key == 114 or key == 82:
            rotation += 10

        # reset on 'n' or 'N'
        elif key == 110 or key == 78:
            rotation = 0
            translation = 0


main()
