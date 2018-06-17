import numpy as np
import cv2
import sys


############################################################
#
#                       KMEANS
#
############################################################


# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    # YOUR CODE HERE
    return np.linalg.norm(a - b)


# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    # YOUR CODE HERE
    # print("update mean")
    pixel_collection = {}
    for k in range(0, len(current_cluster_centers)):
        pixel_collection[k] = []

    for x in range(0, w1):
        for y in range(0, h1):
            cid = clustermask[x][y][0]
            pixel_collection[cid].append(img[x][y])

    for k in pixel_collection.keys():
        # print("current_cluster_centers[k] old: ", current_cluster_centers[k])
        current_cluster_centers[k] = np.uint8(np.mean(pixel_collection[k], axis=0))
        # print("current_cluster_centers[k] new: ", current_cluster_centers[k])


def assign_to_current_mean(img, result, clustermask, mode):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0
    # YOUR CODE HERE
    for x in range(0, w1):
        for y in range(0, h1):
            pixel = img[x][y]
            cluster_color = current_cluster_centers[clustermask[x][y][0]]
            overall_dist += distance(pixel, cluster_color)
            if mode == "quantization":
                result[x][y] = cluster_color                            # Farbquantisierung
            else:
                result[x][y] = cluster_colors[clustermask[x][y][0]]     # k-means basiertes Farbclusterin

    return overall_dist


def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    # YOUR CODE HERE
    for i in range(0, numclusters):
        rnd_x = np.random.randint(0, high=w1)
        rnd_y = np.random.randint(0, high=h1)
        current_cluster_centers[i] = img[rnd_x][rnd_y]


def kmeans(img, mode):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max

    clustermask = np.zeros((h1, w1, 1), np.uint8)
    result = np.zeros((h1, w1, 3), np.uint8)

    # initializes each pixel to a cluster
    # iterate for a given number of iterations or if rate of change is ^very small
    # YOUR CODE HERE
    for i in range(0, max_iter):
        print("iterate:", i, "max:", max_iter, " ", end='')
        changes = 0
        for x in range(0, w1):
            for y in range(0, h1):
                pixel = image[x][y]

                dists = {}
                for j in range(0, len(current_cluster_centers)):
                    dist = distance(current_cluster_centers[j], pixel)
                    dists[j] = dist

                cluster_id = min(dists, key=dists.get)
                if clustermask[x][y][0] != cluster_id:
                    clustermask[x][y] = cluster_id
                    changes += 1

        update_mean(img, clustermask)

        change_rate = changes / (h1 * w1)
        print("change_rate: {:.4f}".format(change_rate), " ", end='')
        over_all_error = assign_to_current_mean(img, result, clustermask, mode)
        print("over all error: {0:.0f}".format(over_all_error))
        if change_rate <= max_change_rate:
            break

    print("k-means done.")
    return result


# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]

results = []
todo = [
    ['./resources/images/Lenna_LAB.png', 3, 'clustering'],
    ['./resources/images/Lenna_LAB.png', 6, 'clustering'],
    ['./resources/images/Lenna_HSV.png', 3, 'clustering'],
    ['./resources/images/Lenna_HSV.png', 6, 'clustering'],
    ['./resources/images/Lenna_RGB.png', 3, 'clustering'],
    ['./resources/images/Lenna_RGB.png', 6, 'clustering'],
    ['./resources/images/Lenna_RGB.png', 0, 'quantization'],
    ['./resources/images/Lenna_RGB.png', 4, 'quantization'],
    ['./resources/images/Lenna_RGB.png', 16, 'quantization'],
    ['./resources/images/Lenna_RGB.png', 32, 'quantization'],
    ['./resources/images/Lenna_RGB.png', 64, 'quantization']
]
for img in todo:

    print("calc img", img[0], "with", img[1], "cluster...")
    # sys.exit()

    # num of cluster
    numclusters = img[1]

    # initialize current cluster centers (i.e. the pixels that represent a cluster center)
    current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

    imgraw = cv2.imread(img[0])
    scaling_factor = 0.5
    imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # compare different color spaces and their result for clustering
    # YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
    image = imgraw
    h1, w1 = image.shape[:2]
    initialize(image)

    # execute k-means over the image
    # it returns a result image where each pixel is color with one of the cluster_colors
    # depending on its cluster assignment
    if numclusters > 0:
        res = kmeans(image, img[2])

        h1, w1 = res.shape[:2]
        h2, w2 = image.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = res
        vis[:h2, w1:w1 + w2] = imgraw

    if numclusters == 0:
        vis = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    if numclusters in [4, 16, 32, 64]:
        vis = cv2.resize(res, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    results.append(vis)


line0 = np.concatenate((results[0], results[1]), axis=1)
line1 = np.concatenate((results[2], results[3]), axis=1)
line2 = np.concatenate((results[4], results[5]), axis=1)
line3 = np.concatenate((results[6], results[7], results[8], results[9], results[10]), axis=1)
scaling_factor = line2.shape[1] / line3.shape[1]
line3 = cv2.resize(line3, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

result = np.concatenate((line0, line1, line2, line3), axis=0)

cv2.imshow("Color-based Segmentation / Kmeans-Clustering", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
