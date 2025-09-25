import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)   # allow 2 hands
offset = 20
imgSize = 300
counter = 0
folder = "Data/Namaste"


while True:
    ret, frame = cap.read()
    hands, frame = detector.findHands(frame)
    cv2.imshow('frame', frame)

    if len(hands) == 2:  # ensure two hands are detected
        hand1, hand2 = hands[0], hands[1]

        # bounding boxes for both hands
        x1, y1, w1, h1 = hand1['bbox']
        x2, y2, w2, h2 = hand2['bbox']

        # combined bounding box (cover both hands)
        x_min = min(x1, x2) - offset
        y_min = min(y1, y2) - offset
        x_max = max(x1 + w1, x2 + w2) + offset
        y_max = max(y1 + h1, y2 + h2) + offset

        # crop both hands together
        imgCrop = frame[y_min:y_max, x_min:x_max]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        if imgCrop.size != 0:  # check crop is valid
            h, w, _ = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow('imgWhite', imgWhite)

            # save on key press
            key = cv2.waitKey(1)
            if key == ord('s'):
                counter += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                print(counter)

    # quit program
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

