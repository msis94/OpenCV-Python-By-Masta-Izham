import cv2
import os
import pandas as pd

# take image from the camera
cam = cv2.VideoCapture(0)

# camera resolution
print('Camera resolution : (width, height) ', cam.get(3), cam.get(4))

# haar-cascade file path
cascade_path = '/home/goblin/Desktop/opencv-haar/haar-file/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cascade_path)

img_dataset = "/home/goblin/Desktop/dataset/"
database_path = '/home/goblin/Desktop/face_database.csv'

# font to be used at rectangle
font = cv2.FONT_HERSHEY_SIMPLEX

# MINIMUM SIZE TO BE RECOGNIZED AS A FACE
minW = int(cam.get(3)/3)
minH = int(cam.get(4)/3)


def capture_data():

    # 1. Employee need to enter their employee number
    emp_num = input('\n Enter Employee Number : ')
    print("\n Please look at the camera and wait.....")

    # 2. Check the existance of the employee number
    df = pd.read_csv(database_path)

    # 3. Get the column of employee number, change it to array and cast it to string
    database = str(df['employee number'].values)

    # 4. If the employee number is exist in database then do the data capturing
    if(emp_num in database):

        # 5. Initialize variable counter to 0, it used to record the current number of the image
        #    been read
        count = 0

        # 6. Using while loop in order to read/extract each frame in video
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
                counter = int(count/max_sample*100)
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.putText(img, str(counter)+"%", (x+5,y+h-5), font, 1, (255,255,0), 2)

                # Save the captured image into the datasets folder
                cv2.imwrite(img_dataset + str(emp_num) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            # To make the window is sizeable
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)

            # To make window open full screen without titlebar
            cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);


            # show rectangle of detected faces
            cv2.imshow('image', img)


            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                # cam.release()
                cv2.destroyAllWindows()
                break
            elif count >= max_sample: # Take 30 face sample and stop video
                # cam.release()
                cv2.destroyAllWindows()
                break

    else:
        print("Employee number does not exist")

while(True):
    max_sample = 30
    choice = int(input('\n Press 1 for Capturing the Image and 2 Recognition : '))

    if(choice==1):
        capture_data()

    else:
        break
        print('mapui')

    # elif(choice==2):
    #     # sepatutnya baut yg ni
    #     # face_recognition()
    #     pass


# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()