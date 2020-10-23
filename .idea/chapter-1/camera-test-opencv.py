'''
- Try catch error (exception handling)
- What if the camera cannot be opened
'''

import numpy as np
import cv2

'''
cv2.VideoCapture(0): Means first camera or webcam.
cv2.VideoCapture(1):  Means second camera or webcam.
cv2.VideoCapture("file name.mp4"): Means video file
'''

cam = cv2.VideoCapture(0)

'''
In this case, if the size of the pixel is to big and we want to reduce it
cam.set(3,640) # set Width
cam.set(4,480) # set Height
'''

# To get the resolution, width(3) and height(4)
width = cam.get(3)
height = cam.get(4)

print('Width of the camera : ', width)
print('Height of the camera : ', height)

while(True):
    '''
    There are two things that can be usefut to be extracted from cam.read()
    1. ret =
    2. frame = image
    '''
    ret, frame = cam.read()
    '''
    In this case we can see the shape of the image which is (480,640,3)
    Bcoz of the channel type, sometime we need to convert it to grayscale
    '''
    print(frame.shape)

    '''
    This one use to flip the position of the image/camera
    cv2.flip(frame, -1) # Flip camera vertically
    cv2.flip(frame, 1) # Flip camera horizontally
    '''

    # To make the window is sizeable
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    # To make window open full screen without titlebar
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cam.release()
cv2.destroyAllWindows()