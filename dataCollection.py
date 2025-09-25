import cv2
import numpy as np
import math
import time


cap=cv2.VideoCapture(0)
from cvzone.HandTrackingModule import HandDetector
detector=HandDetector(maxHands=1)
offset=20
imgSize=300
counter=0
folder="Data/Ok"
while True:
    ret,frame=cap.read()
    hands, frame= detector.findHands(frame)
    cv2.imshow('frame',frame)
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

        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize


        cv2.imshow('imgWhite',imgWhite)


    key=cv2.waitKey(1)
    if key==ord('s'):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)