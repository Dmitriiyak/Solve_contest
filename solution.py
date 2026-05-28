from ultralytics import YOLO
import math
import cv2

img = cv2.imread('run_01/photo/frame1.jpg')
H, W = map(int, img.shape[:2])

model = YOLO('yolo11n.pt')
result = model('run_01/photo/frame1.jpg')
result = result[0]

FOV_x = 65 #!!! РАЗОБРАТЬСЯ
FOV_y = 40 #!!! РАЗОБРАТЬСЯ

fx = W / 2 * math.tan(math.radians(FOV_x / 2))
fy = H / 2 * math.tan(math.radians(FOV_y / 2))

boxes = result.boxes

for i in range(len(boxes)):
        class_id = int(boxes.cls[i].tolist())
        if result.names[class_id] != 'cup':
                continue

        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        print(f"  Рамка: [{x1}, {y1}, {x2}, {y2}]")

        alpha = math.arctan()