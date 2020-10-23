'''
- Try catch error (exception handling)
- What if the haar-cascade file dose not exist
'''


import numpy as np
import cv2

haar_cascade_path = '/home/goblin/Desktop/opencv-haar/haar-file/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(haar_cascade_path)
cam = cv2.VideoCapture(0)
# cap.set(3,640) # set Width
# cap.set(4,480) # set Height

print(cam.get(3), cam.get(4))

width = int(cam.get(3)/3)
height = int(cam.get(4)/3)

print(width, height)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cam.read()
    print('Original Shape (height, width, channel) : ', img.shape)
    #     img = cv2.flip(img, -1)

    # img = cv2.resize(img, (200,200))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, 1.2, 5, minSize=(width, height))

    if len(faces)==0:
        cv2.putText(img, 'FACE NOT DETECTED', (10,40), font, 1, (0,255,255), 2)

    else:
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),10)
            # cv2.rectangle(img,(x,y),(x+100,y+50),(0,255,255),-4)
            # cv2.putText(img, 'FACE DETECTED', (x,y+50), font, 0.7, (255,255,255), 3)
            cv2.putText(img, 'FACE DETECTED', (10,40), font, 1, (255,255,0), 2)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cam.release()
cv2.destroyAllWindows()