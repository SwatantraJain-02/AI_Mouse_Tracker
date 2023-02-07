import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

curTime = 0
prevTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id,lms in enumerate(handLMS.landmark):
                print(id,lms)
                height, width, channel = img.shape
                cx , cy = int(lms.x*width), int(lms.y*height)
                print(id,cx,cy)
                cv2.circle(img,(cx,cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)


    # To calculate FPS and display it
    curTime = time.time()
    fps = 1/(curTime - prevTime)
    prevTime = curTime
    cv2.putText(img,str(int(fps)),(10,70),
                cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)

    # To display image
    cv2.imshow("Image",img)
    cv2.waitKey(1)