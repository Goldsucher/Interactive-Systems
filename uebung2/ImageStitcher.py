# Diese Code gehört noch zu Aufgabe 2 vom Übungsblatt 2
# Stephan Wagner s853668

import numpy as np
import cv2


# complete the class ImageStitcher

class ImageStitcher:
    """A simple class for stitching two images. The class expects two color images"""

    def __init__(self, imagelist):

        self.imagelist = imagelist

        # match ratio to clean feature area from non-important ones
        self.distanceRatio = 0.75
        # theshold for homography
        self.reprojectionThreshold = 4.0

    def match_keypoints(self, kpsPano1, kpsPano2, descriptors1, descriptors2):
        """This function computes the matching of image features between two different
        images and a transformation matrix (aka homography) that we will use to unwarp the images
        and place them correctly next to each other. There is no need for modifying this, we will
        cover what is happening here later in the course.
        """
        # compute the raw matches using a Bruteforce matcher that
        # compares image descriptors/feature vectors in high-dimensional space
        # by employing K-Nearest-Neighbor match (more next course)
        bf = cv2.BFMatcher()
        rawmatches = bf.knnMatch(descriptors1, descriptors2, 2)
        matches = []

        # loop over the raw matches and filter them
        for m in rawmatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. David Lowe's ratio = 0.75)
            # in other words - panorama images need some useful overlap
            if len(m) == 2 and m[0].distance < m[1].distance * self.distanceRatio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # we need to compute a homography - more next course
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsPano1 = np.float32([kpsPano1[i].pt for (_, i) in matches])
            ptsPano2 = np.float32([kpsPano2[i].pt for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsPano1, ptsPano2, cv2.RANSAC, self.reprojectionThreshold)

            # we return the corresponding perspective transform and some
            # necessary status object + the used matches
            return (H, status, matches)

        # otherwise, no homograpy could be computed
        return None

    def draw_matches(self, img1, img2, kp1, kp2, matches, status):
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


    def stitch_to_panorama(self):

        matchList = []
        panoramaImg = []

        current_img = None
        for path in self.imagelist:

            last_img = current_img

            current_img = cv2.imread(path)
            sift = cv2.xfeatures2d.SIFT_create()
            current_kp, current_desc = sift.detectAndCompute(current_img, None)

            if last_img is None:
                # skip first image
                continue
            else:
                last_kp, last_desc = sift.detectAndCompute(last_img, None)

            # 4. match features between the two images consecutive images and check if the
            # result might be None.
            H, status, matches = self.match_keypoints(last_kp, current_kp, last_desc, current_desc)

            # if not enough matches were found we can't stitch
            # and we break here
            if H is None:
                print("Not enough machtes to stitch")
                break

            if len(matches) < 200:
                continue

            result = cv2.warpPerspective(last_img, H, (last_img.shape[1] + current_img.shape[1], last_img.shape[0]))
            result[0:current_img.shape[0], 0:current_img.shape[1]] = current_img

            # 5. create a new image using draw_matches containing the visualized matches
            img_with_matches = self.draw_matches(last_img, current_img, last_kp, current_kp, matches, status)
            matchList.append(img_with_matches)

            current_img = result
            panoramaImg.append(result)

        # 6. return the resulting stitched image
        return (matchList, panoramaImg)
