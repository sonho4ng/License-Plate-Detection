from ultralytics import YOLO

def main():
    absolute_path = r"C:\Users\Admin\LabelImg\windows_v1.8.1\License_Plate_YoloV8_Object_Detection\data.yaml"

    model = YOLO('../yolov8n.pt')

    model.train(data=absolute_path, epochs=1, batch=16, weight_decay=0.001,verbose=True,optimizer='Adam',momentum=0.9,lr0=0.001)

if __name__ == '__main__':
    main()




