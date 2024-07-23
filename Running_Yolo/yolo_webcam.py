from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4, 720)

cap = cv2.VideoCapture('Image/3.mp4')
model = YOLO('runs/detect/train6/weights/last.pt')
classNames = ["plate"]
cn = 1
while True:
    success, img = cap.read()

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # width, height
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2-x1, y2-y1

            # bbox
            cvzone.cornerRect(img,(x1,y1,w,h))
            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            # classname
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{conf} {classNames[cls]}', (max(0,x1), max(30,y1)), scale=0.7, thickness=1,offset=3)





    cv2.imshow("image", img)
    cv2.waitKey(0)
