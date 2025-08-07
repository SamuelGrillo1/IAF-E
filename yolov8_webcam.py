import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

class DetectedObject:
    def __init__(self, center, id):
        self.id = id
        self.center = [center]
        self.frames_seen = 1
        self.last_detected_time = time.time()
        self.disappeared = False
        self.disappeared_once = False

def get_center(box):
    x1, y1, x2, y2 = box[:4]
    return np.array([(x2 + x1) / 2, (y2 + y1) / 2])

def update_objects(objects, detected_boxes, debounce_frames=5):
    current_time = time.time()
    new_objects = {}
    global total_counter

    for box in detected_boxes:
        center = get_center(box)
        obj_id = next((obj_id for obj_id, obj in objects.items() if np.linalg.norm(center - obj.center[-1]) < 50), None)
        if obj_id is None:
            obj_id = max(objects.keys() or [0]) + 1
        obj = objects.get(obj_id, DetectedObject(center, obj_id))
        obj.center.append(center)
        obj.frames_seen += 1
        obj.last_detected_time = current_time
        new_objects[obj_id] = obj

    for obj_id, obj in objects.items():
        if current_time - obj.last_detected_time > debounce_frames:
            obj.disappeared = True
            if not obj.disappeared_once:
                obj.disappeared_once = True
                total_counter += 1

    objects.update(new_objects)
    return objects

def draw_detections(frame, detected_boxes, objects):
    height, width = frame.shape[:2]
    font_scale = 0.75 * (width / 640)
    font_thickness = max(1, int(font_scale * 1.5))

    for obj_id, obj in objects.items():
        if not obj.disappeared:
            last_center = obj.center[-1]
            cv2.circle(frame, (int(last_center[0]), int(last_center[1])), 10, (0, 0, 255), 3)
            for i in range(len(obj.center) - 1):
                cv2.line(frame, (int(obj.center[i][0]), int(obj.center[i][1])), (int(obj.center[i+1][0]), int(obj.center[i+1][1])), (255, 0, 0), 2)

    total_label = f"Total de itens detectados: {total_counter}"
    label_size, _ = cv2.getTextSize(total_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    label_x = (width - label_size[0]) // 2
    label_y = 30
    cv2.rectangle(frame, (label_x, 20), (label_x + label_size[0], label_y + label_size[1]), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, total_label, (label_x, label_y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame

torch.set_default_dtype(torch.float32)
model = YOLO(r'D:\Mills\ProjectMills\IAF&E\IA-Esteira\modelov3\content\runs\detect\train\weights\best.pt')
cap = cv2.VideoCapture(0)  # Default camera
objects = {}
total_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_boxes = results[0].boxes.data.cpu().numpy()
    objects = update_objects(objects, detected_boxes, debounce_frames=5)
    frame = draw_detections(frame, detected_boxes, objects)
    cv2.imshow("Live Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
