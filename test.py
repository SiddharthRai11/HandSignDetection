import cv2
import numpy as np
import math
import pyttsx3

# initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 100)  # speed
engine.setProperty('volume', 1)
engine.startLoop(False)

cap=cv2.VideoCapture(0)
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
detector=HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
offset=20
imgSize=300
counter=0
last_label=""
cooldown=30

labels=["A","B","C"]
while True:
    ret,frame=cap.read()
    frameOutput=frame.copy()
    hands, frame= detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCropShape=imgCrop.shape

        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize
            prediction,index=classifier.getPrediction(imgWhite)
            print(prediction,index)
        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(frameOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 252, 124), 3)
        cv2.rectangle(frameOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        cv2.imshow('imgWhite',imgWhite)

        current_label = labels[index]
        if (current_label != last_label) or (counter == 0):
            engine.say(current_label)
            last_label = current_label
            counter = cooldown
    if counter > 0:
        counter -= 1
    engine.iterate()
    cv2.imshow('frames', frameOutput)
    cv2.waitKey(1)
