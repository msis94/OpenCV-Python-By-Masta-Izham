import cv2
import os
import numpy as np
import time
import pandas as pd
from pathlib import Path
import time

def capture_data():
    # take image from the camera
    cam = cv2.VideoCapture(0)
    print(cam.get(cv2.CAP_PROP_FPS))

    # font to be used at rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX

    # MINIMUM SIZE TO BE RECOGNIZED AS A FACE
    minW = int(cam.get(3)/3)
    minH = int(cam.get(4)/3)

    home = str(Path.home())
    # print(home)
    #------------------------
    # 1. Path of image dataset
    img_dataset = home+'/Desktop/EAFRS/dataset/'

    # 2. Path for haar-cascade pretrained model
    haar_cascade_path = home+'/Desktop/EAFRS/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(haar_cascade_path)

    # 4. Path of employee database
    database = pd.read_csv(home+'/Desktop/EAFRS/face_database.csv')

    max_sample = 30

    # For each person, enter one numeric face id (eg. employee number)
    emp_num = input('\n Enter Employee Number : ')

    databases = str(database['employee number'].values)

    # Initialize individual sampling face count
    count = 0

    if(emp_num in databases):
        print("\n Please look at the camera and wait.....")
        while(True):
            # Take image frame one by one
            ret, img = cam.read()

            # img = cv2.flip(img, -1) # flip video image vertically

            # change from color to grrayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # detect face
            faces = face_detection.detectMultiScale(gray, 1.2, 5, minSize = (minW, minH))

            for (x,y,w,h) in faces:
                count += 1
                counter = count/100*100

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                # cv2.putText(img, str(counter)+"%", (x+5,y+h-5), font, 1, (255,255,0), 2)
                cv2.putText(img, str("{:.2f} %".format(counter)), (x+5,y+h-5), font, 1, (255,255,0), 2)

                # Save the captured image into the datasets folder
                cv2.imwrite(img_dataset + str(emp_num) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])


            # To make the window is sizeable
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)

            # To make window open full screen without titlebar
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

            # show rectangle of detected faces
            cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                cam.release()
                cv2.destroyAllWindows()
                break
            elif count >= max_sample: # Take 30 face sample and stop video
                cam.release()
                cv2.destroyAllWindows()
                break

    else:
        print("Employee number does not exist")

def train_data():

    # font to be used at rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX

    home = str(Path.home())
    # print(home)
    #------------------------
    # 1. Path of image dataset
    img_dataset = home+'/IdeaProjects/OpenCV-Python-By-Masta-Izham/face_database.csv'

    # 2. Path for haar-cascade pretrained model
    haar_cascade_path = home+'/IdeaProjects/OpenCV-Python-By-Masta-Izham/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(haar_cascade_path)

    # 3. Path for LBPH pretrained model
    face_recognition = cv2.face.LBPHFaceRecognizer_create()

    # fisher face
    # face_recognition = cv2.face.FisherFaceRecognizer_create()

    # eigen face
    # face_recognition = cv2.face.EigenFaceRecognizer_create()

    # 4. Trained model path
    trained_model = home+'/Desktop/EAFRS/trainer.yml'

    print("Please wait for model to train")

    img_datasets = [os.path.join(img_dataset,f) for f in os.listdir(img_dataset)]

    # empty array for face samples
    face_sample=[]

    # empy array for ids
    emp_num = []

    count_sample=0
    for image in img_datasets:

        # Read data in grayscale mode
        gray = cv2.imread(image, 0)

        # get the id only from the path name file
        id = int(os.path.split(image)[-1].split(".")[0])
        # print(id)
        faces = face_detection.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            #LBPH
            face_sample.append(gray[y:y+h,x:x+w])
            emp_num.append(id)

        count_sample += 1
        progress = count_sample/len(img_datasets)*100
        print("progress to train {:.2f} %".format(progress))
        #-------------------------------------
    face_recognition.train(face_sample, np.array(emp_num))
    # Save the model into trainer/trainer.yml
    face_recognition.save(trained_model) # recognizer.save() worked on Mac, but not on Pi #asal recognizer.write
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(emp_num))))

def recognize():
    home = str(Path.home())
    # take image from the camera
    cam = cv2.VideoCapture(home+'/IdeaProjects/OpenCV-Python-By-Masta-Izham/ola.webm')

    print(cam.get(cv2.CAP_PROP_FPS))

    # font to be used at rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX

    # MINIMUM SIZE TO BE RECOGNIZED AS A FACE
    minW = int(cam.get(3)/3)
    minH = int(cam.get(4)/3)


    # print(home)
    #------------------------
    # 1. Path of image dataset
    img_dataset = home+'/IdeaProjects/OpenCV-Python-By-Masta-Izham/dataset/'

    # 2. Path for haar-cascade pretrained model
    haar_cascade_path = home+'/IdeaProjects/OpenCV-Python-By-Masta-Izham/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(haar_cascade_path)

    # 3. Path for LBPH pretrained model
    face_recognition = cv2.face.LBPHFaceRecognizer_create()

    # fisher face
    # face_recognition = cv2.face.FisherFaceRecognizer_create()

    # eigen face
    # face_recognition = cv2.face.EigenFaceRecognizer_create()

    # 4. Trained model path
    trained_model = home+'/IdeaProjects/OpenCV-Python-By-Masta-Izham/trainer.yml'

    # 4. Path of employee database
    database = pd.read_csv(home+'/IdeaProjects/OpenCV-Python-By-Masta-Izham/face_database.csv')
    min_acc = 60
    # READ WEIGHT OF LPBH TRAINED MODEL
    face_recognition.read(trained_model)

    prevTime = 0

    while True:
        curTime = time.time()
        start = time.time()
        ret, img =cam.read()
        #     img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = face_detection.detectMultiScale(gray, 1.2, 5, minSize = (minW, minH))

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            #LPBH
            emp_num, confidence = face_recognition.predict(gray[y:y+h,x:x+w])

            # --------------------------------------------
            # find the row of employee number (eg. = 7)
            emp_name = database.loc[database["employee number"]==emp_num]

            # after found employee number show the name of the employee by taking
            # the -name- column
            emp_name = emp_name["name"].values[0]

            acc = round(100 - confidence)

            if (acc > min_acc):
                emp_name
            else:
                emp_name = "unknown"

            # cv2.putText(img, 'Izham Pass : Door Opens', (10,40), font, 1, (255,255,255), 2)
            cv2.putText(img, str(emp_name), (x,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, "{} %".format(str(acc)), (x+5,y+h-5), font, 1, (255,255,0), 1)


        # -------------FPS------------------------
        # curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        fps = 1 / (sec)

        strs = "FPS : %0.1f" % fps

        cv2.putText(img, strs, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # ----------------------------------------

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.imshow('image',img)

        stop = time.time()


        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            print(stop-start)
            cam.release()
            cv2.destroyAllWindows()
            break

# show popup to choose whether to capture/train new data or to recognize

while(True):
    try:
        choice = int(input("\n1. Capture data"
                           "\n2. Train data"
                           "\n3. Recognition face"
                           "\n\nEnter your choice : "))

        if(choice==1):
            capture_data()

        elif(choice==2):
            train_data()

        elif(choice==3):
            recognize()

        elif(choice==4):
            test_accuracy()

        else:
            exit()

    except Exception as e:
        print(e)