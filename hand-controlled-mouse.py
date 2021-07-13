import autopy
from cv2 import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Mediapipe setup
handSol = mp.solutions.hands
hand = handSol.Hands(max_num_hands = 1)
draw = mp.solutions.drawing_utils

screenSize = autopy.screen.size()

while True:
    success, image = webcam.read()
    if success:

        # Process image using mediapipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hand.process(rgb).multi_hand_landmarks

        if result:
            # Move mouse
            landmark = result[0].landmark[0]
            autopy.mouse.move(screenSize[0] * (1 - landmark.x), screenSize[1] * landmark.y)

            # Draw result
            draw.draw_landmarks(image, result[0], handSol.HAND_CONNECTIONS)

        # Display
        cv2.imshow("Hand-Controlled Mouse", image)
        cv2.waitKey(1)