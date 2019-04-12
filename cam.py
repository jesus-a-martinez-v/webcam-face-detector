import argparse
import cv2

from facedetector import FaceDetector
from imutils import resize

argparser = argparse.ArgumentParser()
argparser.add_argument('-f', '--face', required=True, help='Path to where the face cascade resides.')
argparser.add_argument('-v', '--video', required=False, help='Path to the (optional) video file.')
arguments = vars(argparser.parse_args())

face_detector = FaceDetector(arguments['face'])

if not arguments.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(arguments['video'])

while True:
    (grabbed, frame) = camera.read()

    if arguments.get('video') and not grabbed:
        break

    frame = resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rectangles = face_detector.detect(gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30))
    frame_clone = frame.copy()

    green = (0, 255, 0)
    for (f_x, f_y, f_w, f_h) in face_rectangles:
        cv2.rectangle(frame_clone, (f_x, f_y), (f_x + f_w, f_y + f_h), green, 2)

    cv2.imshow('Face', frame_clone)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Did the user press the q key?
        break

camera.release()
cv2.destroyAllWindows()