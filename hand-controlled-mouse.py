from cv2 import cv2

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, image = webcam.read()
    if success:
        cv2.imshow("Hand-Controlled Mouse", image)
        cv2.waitKey(1)