# Blatt 3 "3_tracking"
# Aufgabe 4: Homography und Pose Estimation
# Steffen Burlefinger (859077)

import cv2
import numpy as np

# global constants
min_matches = 10

# initialize flann and SIFT extractor
# note unfortunately in the latest OpenCV + python is a minor bug in the flann
# flann = cv2.FlannBasedMatcher(indexParams, {})
# so we use the alternative but slower Brute-Force Matcher BFMatcher
sift = cv2.xfeatures2d.SIFT_create()
bfMatcher = cv2.BFMatcher()

# extract marker descriptors
img_marker_color = cv2.imread('images/marker.jpg', 1)
kp_marker, descr_marker = sift.detectAndCompute(img_marker_color, None)

def render_virtual_object(img, x_start, y_start, x_end, y_end, quad):
    # define vertices, edges and colors of your 3D object, e.g. cube
    z = 1
    vertices = np.float32([[0, 0, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 1, 0],
                           [0, 0, z],
                           [1, 0, z],
                           [1, 1, z],
                           [0, 1, z]])

    edges = [(0, 1),
             (1, 2),
             (2, 3),
             (3, 0),

             (4, 5),
             (5, 6),
             (6, 7),
             (7, 4),

             (0, 4),
             (1, 5),
             (2, 6),
             (3, 7)]

    color_lines = (0, 0, 0)

    # define quad plane in 3D coordinates with z = 0
    quad_3d = np.float32([[x_start, y_start, 0], [x_end, y_start, 0],
                          [x_end, y_end, 0], [x_start, y_end, 0]])

    h, w = img.shape[:2]
    # define intrinsic camera parameter
    K = np.float64([[w, 0, 0.5 * (w - 1)],
                    [0, w, 0.5 * (h - 1)],
                    [0, 0, 1.0]])

    # find object pose from 3D-2D point correspondences of the 3d quad using Levenberg-Marquardt optimization
    # in order to work we need K (given above and YOUR distortion coefficients from Assignment 2 (camera calibration))
    dist_coef = np.array([0.06489938,  0.2053827,   0.00292677,  0.00300208, -1.12425709])

    # compute extrinsic camera parameters using cv2.solvePnP
    _, rvec, tvec = cv2.solvePnP(quad_3d, quad, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
    # transform vertices: scale and translate form 0 - 1, in window size of the marker
    scale = [x_end - x_start, y_end - y_start, x_end - x_start]
    trans = [x_start, y_start, -x_end - x_start]

    verts = scale * vertices + trans

    # call cv2.projectPoints with verts, and solvePnP result, K, and dist_coeff
    # returns a tuple that includes the transformed vertices as a first argument
    verts, _ = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)

    # we need to reshape the result of projectPoints
    verts = verts.reshape(-1, 2)

    # render edges
    for i, j in edges:
        (x_start, y_start), (x_end, y_end) = verts[i], verts[j]
        cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color_lines, 4)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Interactive Systems: AR Tracking')
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    key = cv2.waitKey(50)
    if key & 0xFF == ord('q'):
        break

    # detect and compute descriptor in camera image
    # and match with marker descriptor
    kp_cam, descr_cam = sift.detectAndCompute(frame, None)
    if descr_marker is None or descr_cam is None:
        print("No Descriptor found")
        continue

    matches = bfMatcher.knnMatch(descr_cam, descr_marker, 2)

    # filter matches by distance [Lowe2004]
    matches = [match[0] for match in matches if len(match) == 2 and
               match[0].distance < match[1].distance * 0.75]

    # if there are less than min_matches we just keep going looking
    # early break
    if len(matches) < min_matches:
        cv2.imshow('Interactive Systems: AR Tracking', frame)
        print("not enough matches")
        continue

    # extract 2d points from matches data structure
    p0 = [kp_marker[m.trainIdx].pt for m in matches]
    p1 = [kp_cam[m.queryIdx].pt for m in matches]

    # transpose vectors
    p0, p1 = np.array([p0, p1])

    # we need at least 4 match points to find a homography matrix
    if len(p0) < 4:
        cv2.imshow('Interactive Systems: AR Tracking', frame)
        print("less than 4 matches points  - no homography matrix found")
        continue

    # find homography using p0 and p1, returning H and status
    # H - homography matrix
    # status - status about inliers and outliers for the plane mapping
    (H, mask) = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)

    # on the basis of the status object we can now filter RANSAC outliers
    mask = mask.ravel() != 0
    if mask.sum() < min_matches:
        cv2.imshow('Interactive Systems: AR Tracking', frame)
        continue

    # take only inliers - mask of Outlier/Inlier
    p0, p1 = p0[mask], p1[mask]

    # get the size of the marker and form a quad in pixel coords np float array using w/h as the corner points
    h1, w1 = img_marker_color.shape[:2]
    quad = [np.array([0, 0], dtype=np.float32),
            np.array([h1, 0], dtype=np.float32),
            np.array([w1, h1], dtype=np.float32),
            np.array([0, w1], dtype=np.float32),]

    print(quad)

    # perspectiveTransform needs a 3-dimensional array
    quad = np.array([quad])
    quad_transformed = cv2.perspectiveTransform(quad, H)

    # transform back to 2D array
    quad = quad_transformed[0]

    # render quad in image plane and feature points as circle using cv2.polylines + cv2.circle
    cv2.polylines(frame, [quad.astype(dtype=np.int)], isClosed=True, color=(0, 0, 255), thickness=2)
    for fp1 in p1:
        fp1 = fp1.astype(np.int)
        cv2.circle(frame, (fp1[0], fp1[1]), 10, (0, 0, 255))

    # render virtual object on top of quad
    render_virtual_object(frame, 0, 0, h1, w1, quad)

    cv2.imshow('Interactive Systems: AR Tracking', frame)

# Destroy Windows and exit the program
cap.release()
cv2.destroyAllWindows()