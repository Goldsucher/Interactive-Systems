# Blatt 2 "2_features"
# Aufgabe 1: SIFT in OpenCV
# Steffen Burlefinger (859077)


import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow('Interactive Systems: Towards AR Tracking')

while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key

    # YOUR CODE HERE

    # Capture frame-by-frame
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()

    # Get the Key Points from the 'gray' image, this returns a numpy array
    kp = sift.detect(img, None)

    # Now we drawn the gray image and overlay the Key Points (kp)
    img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    key = cv2.waitKey(50)
    if key & 0xFF == ord('q'):
        break

    cv2.imshow('Aufgabe 1', img)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()