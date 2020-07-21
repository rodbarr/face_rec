import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_cap = cv2.VideoCapture(0)
img_count = 0

while True:
    ret, frame = video_cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) , 2)
    
    cv2.imshow('FaceDetection', frame)

    if k%256 == 27: #Stop when ESC is pressed
        break
    elif k%256 == 32: #Take a picture when space is pressed
        img_name = "facedetect_webcam_{}.png".format(img_count)
        cv2.imwrite(img_name,frame)
        print("{} written!".format(img_name))
        img_count += 1

video_cap.release()
cv2.destroyAllWindows()