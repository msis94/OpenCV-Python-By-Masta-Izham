import cv2
import numpy as np
import os
import pandas as pd

#

# IMPORT LBPH MODEL
face_recognition = cv2.face.LBPHFaceRecognizer_create()
trained_model_path  = '/home/goblin/Desktop/trainer/trainer.yml'
# READ WEIGHT OF LPBH TRAINED MODEL
face_recognition.read(trained_model_path)

# LOCATION OF HAAR CASCADE FRONTAL FILE
cascade_path = "/home/goblin/Desktop/opencv-haar/haar-file/haarcascade_frontalface_default.xml"
# INITIATE HAAR CASCADE PRETRAINED MODEL
face_detection = cv2.CascadeClassifier(cascade_path);

database_path = "/home/goblin/Desktop/face_database.csv"
# open databese file
database = pd.read_csv(database_path)




# TO EDIT THE FONT USE IN RECTANGLE
font = cv2.FONT_HERSHEY_SIMPLEX

# DECLARATION FOR ID COUNTER
emp_num = 0


# RECORD THE VIDEO USING FIRST CAMERA
cam = cv2.VideoCapture(0)

print(cam.get(3))
print(cam.get(4))

# MINIMUM SIZE TO BE RECOGNIZED AS A FACE
minW = int(cam.get(3)/3)
minH = int(cam.get(4)/3)



while True:
    ret, img =cam.read()
    #     img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray, 1.2, 5, minSize = (minW, minH))

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        emp_num, confidence = face_recognition.predict(gray[y:y+h,x:x+w])

        # --------------------------------------------
        # find the row of employee number (eg. = 7)
        emp_name = database.loc[database["employee number"]==emp_num]

        # after found employee number show the name of the employee by taking
        # the -name- column
        emp_name = emp_name["name"].values[0]


        # --------------------------------------------

        accuracy = round(100 - confidence)


        # Check if confidence is less them 100 ==> "0" is perfect match
        if (accuracy > 60):
            emp_name
        else:
            emp_name = "unknown"

        # cv2.putText(img, 'Izham Pass : Door Opens', (10,40), font, 1, (255,255,255), 2)
        cv2.putText(img, str(emp_name), (x,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, "{} %".format(str(accuracy)), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()