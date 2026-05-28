from ultralytics import YOLO
import math
import cv2

img = cv2.imread('run_01/photo/frame1.jpg')
H, W = map(int, img.shape[:2])

model = YOLO('yolo11n.pt')
result = model('run_01/photo/frame1.jpg')
result = result[0]

boxes = result.boxes

x1 = 0
y1 = 0
x2 = 0
y2 = 0

for i in range(len(boxes)):
        class_id = int(boxes.cls[i].tolist())
        if result.names[class_id] != 'cup':
                continue

        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        print(f"  Рамка: [{x1}, {y1}, {x2}, {y2}]")

FOV_x = 65 #!!! РАЗОБРАТЬСЯ
FOV_y = 40 #!!! РАЗОБРАТЬСЯ

fx = W / (2 * math.tan(math.radians(FOV_x / 2)))
fy = H / (2 * math.tan(math.radians(FOV_y / 2)))
cx = W / 2
x_center = (x1 + x2) / 2
alpha_rad = math.atan2(x_center - cx,  fx)

print(alpha_rad)
print()

import json

def load_lidar(run_path='run_01'):
    points = []
    dist = []
    with open(f'{run_path}/lidar/lidar_scans.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            for i in range(len(data['angles_rad'])):
                angle = data['angles_rad'][i]
                if abs(angle - alpha_rad) < 0.01:
                     points.append(angle)
                     dist.append(data['ranges_m'][i]) # TODO: проверить что не нужно прибавлять 1 из-за null (len должен совпадать у списков)
    return dist

print(load_lidar())