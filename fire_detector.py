from picamera2 import Picamera2, Preview
import time
import cv2
import numpy as np

picam2 = Picamera2()

def configure_cv2():
    cv2.startWindowThread()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

def camera_found_fire():
    configure_cv2()
    while (True):
        picam2.capture_file("test.jpg")
        img = cv2.imread("test.jpg")
        frame = cv2.resize(img, (640, 480))
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = [0, 50, 50]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(hsv, lower, upper)
        _, thr = cv2.threshold(mask, 100, 255, 0)
        conts, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        no_red = cv2.countNonZero(mask)
        res_con = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.drawContours(res_con, conts, -1, (255,0,0), 3)

        if int(no_red) > 50000:
            cv2.imwrite('test.jpg', res_con)
            cv2.destroyAllWindows()

            return True
        return False
