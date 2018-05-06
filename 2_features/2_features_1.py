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

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)

    img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('', img)

    # leave loog on key press 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# clean up
cap.release()
cv2.destroyAllWindows()
