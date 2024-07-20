from ultralytics import YOLO
import cv2
model = YOLO('runs/detect/train6/weights/last.pt')

results = model('Image/3.jpg', show = True)
cv2.waitKey(0)
