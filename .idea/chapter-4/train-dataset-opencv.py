import cv2
import numpy as np
import os
import time

'''
There are 4 path that need to be specify : 
1. Dataset of images path
2. Object (Face) detection pre-trained model (Haar-Casacade) path
3. Object (Face) recognition pre-trained model (LBPH) Path
4. Path of the trained model
'''

start = time.time()

# 1. Path of image dataset
img_dataset = '/home/goblin/Desktop/dataset'

# 2. Path for haar-cascade pretrained model
face_detection = cv2.CascadeClassifier("/home/goblin/Desktop/opencv-haar/haar-file/haarcascade_frontalface_default.xml");

# 3. Path for LBPH pretrained model
face_recognition = cv2.face.LBPHFaceRecognizer_create()

# 4. Trained model path
trained_model = "/home/goblin/Desktop/trainer/trainer.yml"


# function to get the images and label data
def getImagesAndLabels(img_dataset):

    # imagePaths = []
    # for f in os.listdir(img_dataset):
    #     imagePaths.append(os.path.join(img_dataset,f))

    # list comprehension
    img_dataset = [os.path.join(img_dataset,f) for f in os.listdir(img_dataset)]

    # empty array for face samples
    face_sample=[]

    # empy array for ids
    ids = []

    for image in img_dataset:

        # Read data in grayscale mode
        gray = cv2.imread(image, 0)

        # get the id only from the path name file
        id = int(os.path.split(image)[-1].split(".")[0])
        # print(id)
        faces = face_detection.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            face_sample.append(gray[y:y+h,x:x+w])
            ids.append(id)

        #-------------------------------------
    face_recognition.train(face_sample, np.array(ids))


    # Save the model into trainer/trainer.yml
    face_recognition.save(trained_model) # recognizer.save() worked on Mac, but not on Pi #asal recognizer.write
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    return face_sample,ids


print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
face_sample,ids = getImagesAndLabels(img_dataset)
# face_recognition.train(face_sample, np.array(ids))
# # Save the model into trainer/trainer.yml
# face_recognition.save(trained_model) # recognizer.save() worked on Mac, but not on Pi #asal recognizer.write
# # Print the numer of faces trained and end program
# print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

stop = time.time()
time_taken = stop-start
print("Time Taken : ",time_taken)