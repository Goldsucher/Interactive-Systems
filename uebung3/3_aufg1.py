# Diese Code beinhaltet Aufgabe 1 vom Übungsblatt 3
# Der FLANN-Part musste wegen Inkompatiblität mit OpenCV 3.1.0 rausgenommen und durch Funktionen der ImageStitcher.py ersetzt werden
# Stephan Wagner s853668

import cv2
import numpy as np

def match_keypoints(kps1, kps2, desc1, desc2):
    """This function computes the matching of image features between two different
    images and a transformation matrix (aka homography) that we will use to unwarp the images
    and place them correctly next to each other. There is no need for modifying this, we will
    cover what is happening here later in the course.
    """

    # match ratio to clean feature area from non-important ones
    distanceRatio = 0.75
    # theshold for homography
    reprojectionThreshold = 4.0

    # compute the raw matches using a Bruteforce matcher that
    # compares image descriptors/feature vectors in high-dimensional space
    # by employing K-Nearest-Neighbor match (more next course)
    bf = cv2.BFMatcher()
    rawmatches = bf.knnMatch(desc1, desc2, 2)
    matches = []

    # loop over the raw matches and filter them
    for m in rawmatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. David Lowe's ratio = 0.75)
        # in other words - panorama images need some useful overlap
        if len(m) == 2 and m[0].distance < m[1].distance * distanceRatio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # we need to compute a homography - more next course
    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsPano1 = np.float32([kps1[i].pt for (_, i) in matches])
        ptsPano2 = np.float32([kps2[i].pt for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsPano1, ptsPano2, cv2.RANSAC, reprojectionThreshold)

        # we return the corresponding perspective transform and some
        # necessary status object + the used matches
        return (H, status, matches)

def draw_matches(img1, img2, kp1, kp2, matches, status):
    """For each pair of points we draw a line between both images and circles,
    then connect a line between them.
    Returns a new image containing the visualized matches
    """

    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:] = img2

    for ((idx2, idx1), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:

            # x - columns
            # y - rows
            (x1, y1) = kp1[idx1].pt
            (x2, y2) = kp2[idx2].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(vis, (int(x1), int(y1)), 4, (255, 255, 0), 1)
            cv2.circle(vis, (int(x2) + w1, int(y2)), 4, (255, 255, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(vis, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (255, 0, 0), 1)

    return vis

cap = cv2.VideoCapture(0)

#initialize marker image
img_marker_color = cv2.imread('resources/images/marker.jpg', 1)
img_marker_grey = cv2.cvtColor(img_marker_color, cv2.COLOR_BGR2GRAY)

#initialize SIFT
sift = cv2.xfeatures2d.SIFT_create()

#find keypoints and discriptor for marker image
kp_marker, descr_marker = sift.detectAndCompute(img_marker_grey, None)

while(True):

    # Capture frame-by-frame
    ret, img_cam_color = cap.read()
    img_cam_grey = cv2.cvtColor(img_cam_color, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(50)
    if key & 0xFF == ord('q'):
        break

    #find keypoints and discriptor for cam frame
    kp_cam, descr_cam = sift.detectAndCompute(img_cam_grey, None)

    if kp_cam is not None and descr_cam is not None:
        #find matches
        H, status, matches = match_keypoints(kp_marker, kp_cam, descr_marker, descr_cam)

        if status is not None and matches is not None:
            # draw matches
            img_result = draw_matches(img_marker_color, img_cam_color, kp_marker, kp_cam, matches, status)
        elif matches is None:
            print("No matches were found. Program was terminated")
            break
        elif status is None:
            print("No status was found. Program was terminated")
            break

    elif kp_cam is None :
        print("No keypoints were found for cam frame . Program was terminated")
        break
    elif descr_cam is None:
        print("No descriptors were found for cam frame. Program was terminated")
        break

    #show result
    cv2.imshow('Marker / Cam', img_result)


# Destroy Windows and exit the program
cap.release()
cv2.destroyAllWindows()