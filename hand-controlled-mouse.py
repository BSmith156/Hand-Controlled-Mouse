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
            landmarkX = 0
            landmarkY = 0
            for i in [0, 1, 5, 9, 13, 17]:
                landmarkX += result[0].landmark[i].x
                landmarkY += result[0].landmark[i].y
            landmarkX /= 6
            landmarkY /= 6
            current[0] = last[0] + (screenSize[0] * pctBetweenPoints(landmarkX, captureBound[0], captureBound[1]) - last[0]) * smoothing
            if current[0] < 0:
                current[0] = 0
            elif current[0] >= screenSize[0]:
                current[0] = screenSize[0] - 1
            last[0] = current[0]
            current[1] = last[1] + (screenSize[1] * pctBetweenPoints(landmarkY, captureBound[0], captureBound[1]) - last[1]) * smoothing
            if current[1] < 0:
                current[1] = 0
            elif current[1] >= screenSize[1]:
                current[1] = screenSize[1] - 1
            last[1] = current[1]
            autopy.mouse.move(current[0], current[1])
           
            # Left click
            if (result[0].landmark[5].x <= result[0].landmark[4].x <= result[0].landmark[17].x or result[0].landmark[17].x <= result[0].landmark[4].x <= result[0].landmark[5].x):
                autopy.mouse.toggle(down = True)
                cv2.line(image, (int(landmarkX * camSize[0]), int(landmarkY * camSize[1])), (int(result[0].landmark[4].x * camSize[0]), int(result[0].landmark[4].y * camSize[1])), (0, 0, 255), 2)
            else:
                autopy.mouse.toggle(down = False)

            # Draw result
            draw.draw_landmarks(image, result[0], handSol.HAND_CONNECTIONS, draw.DrawingSpec(circle_radius=0))
            cv2.circle(image, (int(landmarkX * camSize[0]), int(landmarkY * camSize[1])), 5, (0, 0, 255), -1)
            cv2.circle(image, (int(result[0].landmark[4].x * camSize[0]), int(result[0].landmark[4].y * camSize[1])), 5, (0, 0, 255), -1)

        # Display
        cv2.rectangle(image, (captureRect[0], captureRect[1]), (captureRect[2], captureRect[3]), (255, 0, 0), 2)
        cv2.imshow("Hand-Controlled Mouse", image)
        cv2.waitKey(1)