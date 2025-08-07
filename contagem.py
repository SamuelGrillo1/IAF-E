import cv2
import os
from ultralytics import YOLO
import torch


torch.set_default_dtype(torch.float32)
torch.set_default_device('cpu')

def process_frame(frame):
    results = model(frame)
    result = results[0]
    detected_boxes = result.boxes.data.cpu().numpy()
    detected_masks = None if result.masks is None else result.masks.data.cpu().numpy()

    item_counter = 0  

    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2, score, class_id = map(int, box)
        label = f"Item-{item_counter}"
        item_counter += 1

        
        if detected_masks is not None:
            mask = detected_masks[i]
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            overlay = frame.copy()
            overlay[mask_resized > 0.5] = segmentation_color[:3]
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        cv2.putText(frame, label, (x1, y1 - 10), text_font, text_size, text_color, text_thickness)

    total_label = f"Total de itens na foto: {item_counter}"
    label_size, _ = cv2.getTextSize(total_label, text_font, text_size * 2, text_thickness)
    label_x = (frame.shape[1] - label_size[0]) // 2
    label_y = label_size[1] + 20
    cv2.rectangle(frame, (label_x, label_y - label_size[1] - 10), (label_x + label_size[0], label_y), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, total_label, (label_x, label_y), text_font, text_size * 2, (255, 255, 255), text_thickness)

    return frame, item_counter


weights_path = 'E:/Mills/ProjectMills\IAF&E/testetrava/content/runs/segment/train2/weights/best.pt'
model = YOLO(weights_path)

folder_path = r'E:\Mills\ProjectMills\IAF&E\testescript'
save_folder = r'E:\Mills\ProjectMills\IAF&E\testescriptResult'
os.makedirs(save_folder, exist_ok=True)

box_color = (2, 64, 195)  # Cor caixa de contorno
text_color = (255, 255, 255)  # Cor do texto
segmentation_color = (33, 112, 243)  # Cor da segmentação
box_thickness = 2  # Espessura caixa de contorno
text_font = cv2.FONT_HERSHEY_SIMPLEX  # Fonte do texto
text_size = 1.0  # Tamanho do texto 
text_thickness = 2  # Espessura do texto 

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    save_path = os.path.join(save_folder, filename)

    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(file_path)
        processed_image, total_items = process_frame(image)
        cv2.imwrite(save_path, processed_image)
        print(f"Imagem processada salva em {save_path}")

    elif filename.endswith('.MOV') or filename.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                processed_frame, _ = process_frame(frame)
                out.write(processed_frame)
            else:
                break

        cap.release()
        out.release()
        print(f"Vídeo processado salvo em {save_path}")
