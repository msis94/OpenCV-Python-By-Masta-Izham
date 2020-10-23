import cv2
import numpy as np
from PIL import Image
import os
import time

start = time.time()
# Path for face image database
# path = '/home/goblin/Desktop/dataset'
path = '/home/goblin/Desktop/dataset'

# path for LBPH pretrained model
recognizer = cv2.face.LBPHFaceRecognizer_create()

# path for haar-cascade pretrained model
detector = cv2.CascadeClassifier("/home/goblin/Desktop/opencv-haar/haar-file/haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    # empty array for face samples
    faceSamples=[]

    # empy array for ids
    ids = []

    for imagePath in imagePaths:
        # print(imagePath)
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.save('/home/goblin/Desktop/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi #asal recognizer.write
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

stop = time.time()
time_taken = stop-start
print("Time Taken : ",time_taken)