# Diese Code beinhaltet Aufgabe 3 vom Übungsblatt 2
# Stephan Wagner s853668

import cv2
import glob
import numpy as np
import math
from queue import PriorityQueue

############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
def distance(a, b):
    return np.linalg.norm(a - b)


def create_keypoints(w, h):
    keypoints = []
    keypoint_size = 11   # has to be odd

    radius = int(keypoint_size/2)
    step = radius*2

    for cx in range(radius, w, step):
        for cy in range(radius, h, step):
            keypoints.append(cv2.KeyPoint(cx, cy, keypoint_size))

    return keypoints


# 1. preprocessing and load
images = glob.glob('./resources/images/database/*/*.jpg')

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
key_points = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
print("Process is running")
sift = cv2.xfeatures2d.SIFT_create()
for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(".", end="", flush=True)
    key_points, descriptor = sift.compute(gray, key_points)
    descriptors.append((descriptor, img_path))

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())
result = None
for query_img_path in ['./resources/images/database/query_face.jpg',
                       './resources/images/database/query_car.jpg',
                       './resources/images/database/query_flower.jpg']:

    query_img = cv2.imread(query_img_path)
    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    key_points, descriptor = sift.compute(gray, key_points)

    q = PriorityQueue()
    for img_db in descriptors:
        print(".", end="", flush=True)
        d = distance(descriptor, img_db[0])
        q.put((d, img_db[1]))

    # 5. output (save and/or display) the query results in the order of smallest distance
    imgs = []
    while not q.empty():
        v, img = q.get()
        imgs.append(cv2.imread(img))

    img_left = query_img
    for i in range(0, math.ceil(len(imgs)/2)):
        if len(imgs) % 2 == 1 and i == int(len(imgs)/2):
            img_top = np.full_like(imgs[0], 255)
        else:
            img_top = imgs[i]
        img_right = np.concatenate((img_top, imgs[i+int(len(imgs)/2)]), axis=0)
        img_right = cv2.resize(img_right, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        img_left = np.concatenate((img_left, img_right), axis=1)

    if result is None:
        result = img_left
    else:
        result = np.concatenate((result, img_left), axis=0)

print("\nProcess is finished")
cv2.imshow('', result)
key = cv2.waitKey()
cv2.destroyAllWindows()
