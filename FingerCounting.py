# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:57:13 2022

@author: ozlemakgunoglu
"""

import cv2
import time
import os
import handTrackingModule as htm


wCam, hCam=640,480

cap = cv2.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)

folderPath="FingerImages"
myList=os.listdir(folderPath)
print(myList)
overlayList=[]

#impath jpg
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    #kaydetmek için
    overlayList.append(image)
# print(len(overlayList))    

pTime=0


#detectionConfidence 0.75
detector=htm.handDetector(detectionCon=0.75)

#4=thumb, 8=index , 12= middle , 16 = ring , 20= pinkey
tipIds = [ 4, 8, 12, 16, 20]



while cap.isOpened():
    
    success , img=cap.read()
    img= detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    # print(lmlist)
    
    if len(lmlist) !=0:
        #parmak kapalı mı açık mı tespit edelim 
        #8. nokta nın 2. ekseni yani y ekseni değeri
        # if lmlist[8][2]<lmlist[6][2]:
        #     print("Index finger open")
        fingers = []
        
        #for right hands thumb
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        
        #other 4  fingers
        for id in range(1,5):
            if lmlist[tipIds[id]][2]<lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # print(fingers)
        #Count() listede kaç tane 1 var sayar Count(1)
        totalFingers = fingers.count(1)
        print(totalFingers)
                
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]
        
        #başlangıç noktası 20,225 bitiş noktası 170,425
        cv2.rectangle(img,(20,320),(170,425) , (0,255,0),cv2.FILLED )
        cv2.putText(img,str(totalFingers),(60,410),cv2.FONT_HERSHEY_PLAIN,7,(255,0,0),20 )
    cTime = time.time() #current time
    fps = 1/(cTime-pTime) #previous time
    pTime = cTime
    
    cv2.putText(img,f'FPS:{int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3 )
    #location 400.70  , scale 3 ,thickness 3 
    
    
    cv2.imshow("Camera",img)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()    
cv2.destroyAllWindows()