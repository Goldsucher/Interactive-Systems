import cv2


def main():
    cap = cv2.VideoCapture(0)
    ch = 0

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # wait for key and switch to mode
        key = cv2.waitKey(1) & 0xFF
        ch = key if key != 255 else ch

        if ch == ord('1'):
            # to Hue Saturation Value | code taken from https://goo.gl/iEVmzG (opencv-python-tutroals.readthedocs.io)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if ch == ord('2'):
            # to CIEL*a*b* | code taken from https://goo.gl/eEbxsp (stackoverflow.com)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        if ch == ord('3'):
            # to YUV | code taken from https://goo.gl/9N1wse (opencv.org)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        if ch == ord('4'):
            # Adaptives Thresholding: Gaussian - Thresholding | code taken from https://goo.gl/C6fniq (opencv.org)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.medianBlur(frame, 5)
            frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if ch == ord('5'):
            # Adaptives Thresholding: Otsu - Thresholding | code taken from https://goo.gl/C6fniq (opencv.org)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if ch == ord('6'):
            # Canny-Edge | code taken from https://goo.gl/69MuAt (opencv.org)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.Canny(frame, 100, 200)

        if ch == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


main()
