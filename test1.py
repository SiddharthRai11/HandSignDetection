import cv2
import numpy as np
import math
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 100)
engine.setProperty('volume', 1)
engine.startLoop(False)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
counter = 0
last_label = ""
cooldown = 30
labels = ["A", "B", "C", "Help", "Ok", "ThankYou", "Yes"]

def preprocess_and_predict(frame, bbox):
    x, y, w, h = bbox
    imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    aspectRatio = h / w

    if aspectRatio > 1:  # tall
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:  # wide
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hCal + hGap, :] = imgResize

    prediction, index = classifier.getPrediction(imgWhite)
    return labels[index], imgWhite

while True:
    ret, frame = cap.read()
    frameOutput = frame.copy()
    hands, frame = detector.findHands(frame)

    if len(hands) == 2:   # --- Two hands detected (combined gesture) ---
        hand1, hand2 = hands

        # Get bounding box that covers both hands
        x1, y1, w1, h1 = hand1['bbox']
        x2, y2, w2, h2 = hand2['bbox']

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)

        combined_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Predict using combined crop
        combined_label, imgWhiteCombined = preprocess_and_predict(frame, combined_bbox)

        # Draw combined box + label
        cv2.putText(frameOutput, combined_label, (x_min, y_min - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 252, 124), 3)
        cv2.rectangle(frameOutput,
                      (x_min - offset, y_min - offset),
                      (x_max + offset, y_max + offset),
                      (255, 0, 255), 4)

        cv2.imshow("Two Hands Combined", imgWhiteCombined)

        # Speak the combined label
        if (combined_label != last_label) or (counter == 0):
            engine.stop()
            engine.say(combined_label)
            last_label = combined_label
            counter = cooldown

    elif len(hands) == 1:   # --- Only one hand detected ---
        hand = hands[0]
        label, imgWhite = preprocess_and_predict(frame, hand['bbox'])
        x, y, w, h = hand['bbox']
        cv2.putText(frameOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 252, 124), 3)
        cv2.rectangle(frameOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.imshow("One Hand", imgWhite)

        if (label != last_label) or (counter == 0):
            engine.stop()
            engine.say(label)
            last_label = label
            counter = cooldown

    if counter > 0:
        counter -= 1

    engine.iterate()
    cv2.imshow("frame", frameOutput)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

