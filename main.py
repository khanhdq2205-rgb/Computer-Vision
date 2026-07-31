# Import library
import cv2 
import numpy as np

# Create face_cascade object 
face_cascade_path = "Haar Cascade Classifier\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Cannot create object")
    exit()

# Open webcam and check for open 
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open the camera")
    exit()

# Read the camera and draw rectangle
while True:
    cap, frame = cam.read()
    if not cap:
        print("Cannot read the camera")
        exit()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rect = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=3)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    
    frame = cv2.flip(frame, 1)
    cv2.imshow("Face detection", frame)

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

cv2.release()
cv2.destroyAllWindow()



