import autopy.screen
import cv2
import numpy as np
import HandDetectionModule as hdm
import time

##########################################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
##########################################

cTime, pTime = 0, 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = hdm.HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # 1. Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and the middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print("1st->", x1, y2, "2nd->", x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # print(fingers)
        # 4. Only Index Finger : It is in moving mode
        if (fingers[1] == 1) and fingers[2] == 0:
            # 5. Convert our coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 7, (255, 127, 100), cv2.FILLED)
            plocX, plocY = clocX, clocY

            # 8. Both Index and middle fingeres are up : It is in clicking mode
        if (fingers[1] == 1) and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # 10. Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 7, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    # 11. Frame Rate
    cTime = time.time();

    fps = 1 / (cTime - pTime)
    pTime = cTime
    # 12. Display
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
