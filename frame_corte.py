import cv2
import os
import imageio

# Caminho onde os vídeos estão armazenados
VIDEO_PATH = r"D:\Mills\ProjectMills\Kaizen8\dataset\videos"

# Caminho onde as imagens extraídas serão salvas
SAVE_PATH = r"D:\Mills\ProjectMills\Kaizen8\dataset\frames"

# Certifique-se de que o diretório de destino existe
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Lista de todos os vídeos no diretório
videos = [os.path.join(VIDEO_PATH, video) for video in os.listdir(VIDEO_PATH) if video.endswith(('.mp4', '.MOV'))]

frame_counter = 0
for video in videos:
    print(f"Processando vídeo: {video} -----")
    
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte de BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Salva o frame a cada segundo
        if frame_counter % fps == 0:
            filename = os.path.join(SAVE_PATH, f"frame_{frame_counter}.jpg")
            try:
                imageio.imsave(filename, frame_rgb)
                print(f"Salvando frame {frame_counter} no caminho: {filename}")
            except Exception as e:
                print(f"Erro ao salvar frame no caminho: {filename}. Erro: {str(e)}")
        
        frame_counter += 1
        
    cap.release()

print("Processamento concluído!")
