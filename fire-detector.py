from picamera2 import Picamera2
import time
import cv2
import numpy as np

Fire_Reported = 0

cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

while True:
    picam2.capture_file("test.jpg")
    img = cv2.imread("test.jpg")
    frame = cv2.resize(img, (640, 480))
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
        Fire_Reported = Fire_Reported + 1

    cv2.imshow("output", output)

    if (Fire_Reported >= 1):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(Fire_Reported)
cv2.destroyAllWindows()
