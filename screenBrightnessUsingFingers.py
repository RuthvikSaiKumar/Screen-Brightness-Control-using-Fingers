import time
import screen_brightness_control as sbc
import cv2 as cv
import mediapipe as mp
import math
import numpy as np

cap = cv.VideoCapture(0)
cap.set(3, 750)
cap.set(4, 600)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence = 0.8, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            points = [[] for _ in range(21)]
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                points[id]=[cx, cy]
                if id in [4, 8]:
                    cv.circle(img, (cx,cy), 15, (255,0,255), cv.FILLED)
            
            if len(points) != 0:
                x1, y1 = points[4][0], points[4][1]
                x2, y2 = points[8][0], points[8][1]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
                cv.circle(img, (cx,cy), 15, (255,0,255), cv.FILLED)
                
                length = math.hypot(x2-x1, y2-y1)
                
                if length < 40:
                    cv.circle(img, (cx,cy), 15, (0, 255, 0), cv.FILLED)
                    
                minBrightness = 1
                maxBrightness = 100
                # min range = 40
                # max range = 200
                
                brightness = np.interp(length, [40, 200], [minBrightness, maxBrightness])
                sbc.set_brightness(brightness)
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv.putText(img, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    cv.imshow('frame', img)
    if cv.waitKey(1) == ord('q'):
        break
