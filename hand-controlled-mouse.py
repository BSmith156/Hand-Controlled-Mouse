import autopy
from cv2 import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
flip = True

smoothing = 0.2
current = [0, 0]
last = [0, 0]

screenSize = autopy.screen.size()
camSize = (webcam.get(cv2.CAP_PROP_FRAME_WIDTH), webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

captureArea = 0.5
captureBound = ((1 - captureArea) / 2, 1 - ((1 - captureArea) / 2))
captureRect = (int((camSize[0] / 2) - (camSize[0] * captureArea / 2)), int((camSize[1] / 2) - (camSize[1] * captureArea / 2)), int((camSize[0] / 2) + (camSize[0] * captureArea / 2)), int((camSize[1] / 2) + (camSize[1] * captureArea / 2)))

def pctBetweenPoints(point, min, max):
    return (point - min) / (max - min)

# Mediapipe setup
handSol = mp.solutions.hands
hand = handSol.Hands(max_num_hands = 1)
draw = mp.solutions.drawing_utils

while True:
    success, image = webcam.read()

    if flip:
        image = cv2.flip(image, 1)

    if success:

        # Process image using mediapipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hand.process(rgb).multi_hand_landmarks

        if result:
            # Move mouse
            landmark = result[0].landmark[0]
            current[0] = last[0] + (screenSize[0] * pctBetweenPoints(landmark.x, captureBound[0], captureBound[1]) - last[0]) * smoothing
            current[1] = last[1] + (screenSize[1] * pctBetweenPoints(landmark.y, captureBound[0], captureBound[1]) - last[1]) * smoothing
            try:
                autopy.mouse.move(current[0], current[1])
            except ValueError:
                pass
            last[0] = current[0]
            last[1] = current[1]

            # Draw result
            draw.draw_landmarks(image, result[0], handSol.HAND_CONNECTIONS)

        # Display
        cv2.rectangle(image, (captureRect[0], captureRect[1]), (captureRect[2], captureRect[3]), (255, 0, 0), 2)
        cv2.imshow("Hand-Controlled Mouse", image)
        cv2.waitKey(1)