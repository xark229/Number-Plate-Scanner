import cv2
import os

cap=cv2.VideoCapture(0)
nPlateCasscade=cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
min_ar=500

cap.set(3,640)
cap.set(4,480)
cap.set(10,50)
count=len(os.listdir("scanned"))

while True:
    success,img=cap.read()
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate = nPlateCasscade.detectMultiScale(imggray, 1.1, 2)
    for (x, y, w, h) in nplate:
        area=w*h
        if area>min_ar:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 150, 0), 2)
            cv2.putText(img,"Number plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,150,0),1)
            imgRes=img[y:y+h,x:x+w]
            cv2.resize(imgRes,(500,350))
            cv2.imshow("number palte", imgRes)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF==ord('s'):
        cv2.imwrite("scanned/No_plate"+str(count)+".jpg",imgRes)
        cv2.rectangle(img,(0,200),(640,300),(255,0,255),cv2.FILLED)
        cv2.putText(img,"Saved successfully.....",(320,210),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,0),2)
        cv2.imshow("Video",img)
        cv2.waitKey(500)
        count+=1
    elif cv2.waitKey(1) & 0xFF==ord('q'):
        break
    else: continue

cv2.destroyWindow()