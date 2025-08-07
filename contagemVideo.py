import cv2
import os
from ultralytics import YOLO
import torch
import numpy as np

def get_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def process_frame(frame, recent_detections):
    results = model(frame)
    result = results[0]
    detected_boxes = result.boxes.data.cpu().numpy()

    item_counter = len(recent_detections)
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2, score, class_id = map(int, box)

        if score < 0.5:  # Ignora detecções com baixa confiança
            continue

        new_detection = True
        for recent_box in recent_detections:
            if get_iou((x1, y1, x2, y2), recent_box) > 0.5:
                new_detection = False
                break

        if new_detection:
            recent_detections.append((x1, y1, x2, y2))
            item_counter += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        if new_detection:
            label = f"Item-{item_counter - 1}"
            cv2.putText(frame, label, (x1, y1 - 10), text_font, text_size, text_color, text_thickness)

    # Limpa detecções antigas
    if len(recent_detections) > 50:
        recent_detections = recent_detections[-50:]

    total_label = f"Total de itens detectados: {item_counter}"
    label_size, _ = cv2.getTextSize(total_label, text_font, text_size * 2, text_thickness)
    label_x = (frame.shape[1] - label_size[0]) // 2
    label_y = label_size[1] + 20
    cv2.rectangle(frame, (label_x, label_y - label_size[1] - 10), (label_x + label_size[0], label_y), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, total_label, (label_x, label_y), text_font, text_size * 2, (255, 255, 255), text_thickness)

    return frame, item_counter, recent_detections

torch.set_default_tensor_type('torch.FloatTensor')
weights_path = 'E:/Mills/ProjectMills\IAF&E/testetrava/content/runs/segment/train2/weights/best.pt'
model = YOLO(weights_path)

folder_path = r'E:\Mills\ProjectMills\IAF&E\testescript'
save_folder = r'E:\Mills\ProjectMills\IAF&E\testescriptResult'
os.makedirs(save_folder, exist_ok=True)

box_color = (2, 64, 195)
text_color = (255, 255, 255)
box_thickness = 2
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_size = 1.0
text_thickness = 2

recent_detections = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    save_path = os.path.join(save_folder, filename)


    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(file_path)
        processed_image, total_items, recent_detections = process_frame(image, recent_detections)
        cv2.imwrite(save_path, processed_image)
        print(f"Imagem processada salva em {save_path}")

    elif filename.endswith('.MOV') or filename.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                processed_frame, _, recent_detections = process_frame(frame, recent_detections)
                out.write(processed_frame)
            else:
                break

        cap.release()
        out.release()
        print(f"Vídeo processado salvo em {save_path}")
