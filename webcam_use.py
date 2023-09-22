# CODE FOR EXTERNAL WEBCAM

import cv2  # install OpenCV
import mediapipe as mp  # mediapipe
from pynput.keyboard import Controller
from time import sleep
import math

# import numpy as np  (till now numpy installation not required)

# Capturing video from camera
cap = cv2.VideoCapture(1)  # value 0 for laptop camera, 1 for external webcam attach to pc
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Initialize mediapipe hand class
mpHands = mp.solutions.hands
# Set up the Hands function
Hands = mpHands.Hands()
# Initialize the mediapipe drawing class
mpDraw = mp.solutions.drawing_utils

keys = ["W"]  # Word to be displayed
keyboard = Controller()


class Store():
    def __init__(self, pos, size, text):
        self.pos = pos
        self.size = size
        self.text = text


# Draw function for displaying key with grey rectangle box
def draw(img, storedVar):
    for button in storedVar:
        x, y = button.pos = (600, 280)
        w, h = button.size = (100, 100)
        cv2.rectangle(img, button.pos, (x + w, y + h), (64, 64, 64), 2)
        cv2.putText(img, button.text, (x + 15, y + 80), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255), 2)
    return img


StoredVar = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        StoredVar.append(Store([60 * j + 10, 60 * i + 10], [50, 50], key))

# Checking camera is opened or not
while cap.isOpened():
    success_, img = cap.read()  # reading Frame
    cvtImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting BGR to RGB
    results = Hands.process(cvtImg)  # Processing image for tracking

    lmList = []  # Initialize a list to store the detected landmarks of the hand.

    if results.multi_hand_landmarks:  # Getting landmarks (location) of hands if exists
        for img_in_frame in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,  # image to draw
                                  img_in_frame,  # model output
                                  mpHands.HAND_CONNECTIONS,  # hand connections
                                  mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                  mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                     circle_radius=2))  # Drawing hands connection
            # Iterate over the found hands.
        for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([cx, cy])  # Append the landmark into the list.

    if lmList:
        for button in StoredVar:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][
                1] < y + h:  # 8th element of the hand (tip point of index finger)
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255),
                              cv2.FILLED)  # if 8th point reach the near the box draw red rectangle
                # Get the length
                x1, y1 = lmList[8][0], lmList[8][1]
                x2, y2 = lmList[12][0], lmList[12][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)  # Calculate the distance between the two element
                print(l)  # Print the distance
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[12][0] < x + w and y < lmList[12][
                1] < y + h:  # 12th element of the hand (tip point of index finger)
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[12][0], lmList[12][1]
                x2, y2 = lmList[16][0], lmList[16][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[16][0] < x + w and y < lmList[16][
                1] < y + h:  # 16th element of the hand (tip point of index finger)
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[16][0], lmList[16][1]
                x2, y2 = lmList[20][0], lmList[20][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[20][0] < x + w and y < lmList[20][
                1] < y + h:  # 20th element of the hand (tip point of index finger)
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[20][0], lmList[20][1]
                x2, y2 = lmList[4][0], lmList[4][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[4][0] < x + w and y < lmList[4][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[4][0], lmList[4][1]
                x2, y2 = lmList[12][0], lmList[12][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[7][0] < x + w and y < lmList[7][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[7][0], lmList[7][1]
                x2, y2 = lmList[11][0], lmList[11][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[11][0] < x + w and y < lmList[11][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[11][0], lmList[11][1]
                x2, y2 = lmList[15][0], lmList[15][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[15][0] < x + w and y < lmList[15][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[15][0], lmList[15][1]
                x2, y2 = lmList[19][0], lmList[19][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[19][0] < x + w and y < lmList[19][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[19][0], lmList[19][1]
                x2, y2 = lmList[3][0], lmList[3][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[3][0] < x + w and y < lmList[3][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[3][0], lmList[3][1]
                x2, y2 = lmList[7][0], lmList[7][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)

            if x < lmList[6][0] < x + w and y < lmList[6][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[6][0], lmList[6][1]
                x2, y2 = lmList[10][0], lmList[10][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[10][0] < x + w and y < lmList[10][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[10][0], lmList[10][1]
                x2, y2 = lmList[14][0], lmList[14][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[14][0] < x + w and y < lmList[14][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[14][0], lmList[14][1]
                x2, y2 = lmList[18][0], lmList[18][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[18][0] < x + w and y < lmList[18][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[18][0], lmList[18][1]
                x2, y2 = lmList[2][0], lmList[2][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[2][0] < x + w and y < lmList[2][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[2][0], lmList[2][1]
                x2, y2 = lmList[6][0], lmList[6][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[5][0] < x + w and y < lmList[5][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[5][0], lmList[5][1]
                x2, y2 = lmList[9][0], lmList[9][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[9][0] < x + w and y < lmList[9][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[9][0], lmList[9][1]
                x2, y2 = lmList[13][0], lmList[13][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[13][0] < x + w and y < lmList[13][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[13][0], lmList[13][1]
                x2, y2 = lmList[17][0], lmList[17][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[17][0] < x + w and y < lmList[17][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[17][0], lmList[17][1]
                x2, y2 = lmList[1][0], lmList[1][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[1][0] < x + w and y < lmList[1][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[1][0], lmList[1][1]
                x2, y2 = lmList[5][0], lmList[5][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
            if x < lmList[0][0] < x + w and y < lmList[0][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                x1, y1 = lmList[0][0], lmList[0][1]
                x2, y2 = lmList[17][0], lmList[17][1]
                l = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
                print(l)
                ## when clicked
                if not l > 63:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                    sleep(0.15)
    img = draw(img, StoredVar)  # Draw the key and box

    cv2.imshow("Hand Tracking", img)   # Display the frame

    if cv2.waitKey(1) == 113:  # Q=113 it will stop and exit
        break
