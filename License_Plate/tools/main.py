from ultralytics import YOLO
import cv2
import cvzone
import util
import torch
from License_Plate.sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('../../yolov8n.pt')
license_plate_detector = YOLO('../../Running_Yolo/runs/detect/train6/weights/last.pt')

# load video
cap = cv2.VideoCapture('../../Running_Yolo/Image/4.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True


while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        print('vid err')
        continue
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cvzone.putTextRect(frame,'vehicles', (max(0, int(x1)), max(30, int(y1))), scale=0.7, thickness=1,
                                  offset=3)

        # track vehicles
        detections_array = np.asarray(detections_)
        if detections_array.size == 0:
            pass
        else:
            track_ids = mot_tracker.update(detections_array)
        # detect license plates
        x =0
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            x +=1
            print('Number of license_plate detected: ', x)
            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                print(license_plate_crop.shape)
                # read license plate number
                if license_plate_crop is not None and license_plate_crop.size > 0:

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
                    if license_plate_text is None:
                        license_plate_text = 'undefin'
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cvzone.putTextRect(frame, license_plate_text, (max(0, int(x1)), max(30, int(y1))), scale=2,
                                       thickness=2,
                                       offset=3)
                # cv2.resize(license_plate_crop, (1000,680))

                # cv2.imshow('crop', license_plate_crop)
                # cv2.imshow('frame', license_plate_crop)
                # cv2.waitKey(15)

                print(license_plate_text)



                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': 0}}

    # device = next(license_plate_detector.parameters()).device
    # print(f'Model is running on device: {device}')
    frame = cv2.resize(frame, (1260,960))
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
# write results
write_csv(results, './test.csv')