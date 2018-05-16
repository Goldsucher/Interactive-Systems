# Diese Code beinhaltet Aufgaben 1 vom Übungsblatt 2
# Stephan Wagner s853668

import cv2

def siftDetection(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)

    img_sift = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_sift

cap = cv2.VideoCapture(0)

mode = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(50)
    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('s'):
        if mode == 1:
            mode = 0
        else:
            mode = 1

    if mode == 1:
        img = siftDetection(img)

    cv2.imshow('frame', img)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()