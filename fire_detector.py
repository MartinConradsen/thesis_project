from picamera2 import Picamera2, Preview
import time
import cv2
import numpy as np

picam2 = Picamera2()

def configure_cv2():
    picam2.stop()
    cv2.destroyAllWindows()
    cv2.startWindowThread()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

def camera_found_fire():
    picam2.capture_file("test.jpg")
    img = cv2.imread("test.jpg")
    frame = cv2.resize(img, (960, 540))

    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)

    if int(no_red) > 15000:
        cv2.imwrite('test.jpg', output)
        cv2.imshow("output", output)
        cv2.destroyAllWindows()

        return True
    return False
