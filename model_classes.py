from ultralytics import YOLO
model = YOLO('./models/yolov8/Handball_SideView.pt')
print(model.names)  # This will show the class names