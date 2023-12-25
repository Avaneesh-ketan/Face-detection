import cv2 as cv
import time
import mediapipe as mp

cap = cv.VideoCapture(0)
cTime = 0
pTime = 0
mpFace = mp.solutions.face_detection #extracting Fdace Detection modules from mediapipe
face = mpFace.FaceDetection() #utilising the function of Face Detection
mpDraw = mp.solutions.drawing_utils #drawing on the video for landmarks
while True:
    suc, img = cap.read()
    imgRBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face.process(img)
    if results.detections:
        for id,detection in enumerate(results.detections):
            print(id, detection)
            print(detection.score)
            #bounding box is the box which covers the human faces in the detection
            print(detection.location_data.relative_bounding_box)
            mpDraw.draw_detection(img, detection)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv.imshow("Image", img)
    cv.waitKey(1)