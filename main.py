import os
import cv2
import torch
import numpy as np
import pygame
import smtplib
from email.message import EmailMessage

#######################################
# Configuración del correo electrónico
#######################################
# Modifica estos valores con tus datos
EMAIL_SENDER = 'jleitonarias@gmail.com'
EMAIL_PASSWORD = 'vjre pqzf qoga kjqc'  # O contraseña de aplicación
EMAIL_RECEIVER = 'jleitonarias@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465


def send_email(photo_path):
    subject = 'Alerta: Persona detectada'
    body = 'Se ha detectado una persona en la zona de interés. Adjunto la foto de la detección.'
    
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content(body)
    
    with open(photo_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(photo_path)
    
    # Adjuntar la imagen
    msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)
    
    # Enviar el correo a través de SMTP_SSL
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)
        print("Correo enviado a", EMAIL_RECEIVER)

#######################################
# Configuración del entorno y modelo
#######################################
# Ruta donde se guardarán las fotos detectadas
save_folder = "/Users/developerkrika/Desktop/Detected_Photos"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Ruta al sonido de alarma
path_alarm = "Alarm/alarm.wav"

pygame.init()
pygame.mixer.music.load(path_alarm)

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Configuración de la fuente de video: usa la webcam en tiempo real si use_webcam es True;
# en caso contrario, se utiliza un video de archivo.
use_webcam = False  # Cambia a False si deseas usar un video pregrabado
if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("Test Videos/thief_video2.mp4")

# Clases objetivo (se pueden incluir otras, aquí se usa person para la notificación)
target_classes = ['car', 'bus', 'truck', 'person']

count = 0
number_of_photos = 3

# Lista para almacenar los puntos del polígono (zona de interés)
pts = []

#######################################
# Funciones para el manejo del polígono
#######################################
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Punto agregado:", x, y)
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []
        print("Polígono reiniciado.")

def inside_polygon(point, polygon):
    # Acepta el punto si está dentro o sobre el borde del polígono
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    return result >= 0

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

#######################################
# Función de preprocesado
#######################################
def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

#######################################
# Bucle principal
#######################################
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Finaliza si no hay más frames
    
    # Preprocesar el frame y crear una copia para el recorte sin overlays
    frame_processed = preprocess(frame)
    frame_copy = frame_processed.copy()
    
    results = model(frame_processed)
    
    # Recorrer cada objeto detectado
    for index, row in results.pandas().xyxy[0].iterrows():
        center_x = None
        center_y = None
        
        if row['name'] in target_classes:
            name = str(row['name'])
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Dibujar el bounding box, nombre y centro en el frame
            cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame_processed, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame_processed, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Si se ha definido el polígono y se detecta una persona
            if len(pts) >= 4 and name == 'person':
                if inside_polygon((center_x, center_y), np.array(pts)):
                    # Ajustar el bounding box con un margen para incluir algo de contexto
                    margin = 10
                    x1_adj = max(x1 - margin, 0)
                    y1_adj = max(y1 - margin, 0)
                    x2_adj = min(x2 + margin, frame_processed.shape[1])
                    y2_adj = min(y2 + margin, frame_processed.shape[0])
                    
                    # Recortar la imagen de la persona detectada
                    person_crop = frame_copy[y1_adj:y2_adj, x1_adj:x2_adj]
                    
                    # Guardar la imagen y enviar el correo si no se ha alcanzado el límite
                    if count < number_of_photos:
                        save_path = os.path.join(save_folder, "detected" + str(count) + ".jpg")
                        cv2.imwrite(save_path, person_crop)
                        print("Imagen guardada en:", save_path)
                        
                        try:
                            send_email(save_path)
                        except Exception as e:
                            print("Error al enviar el correo:", e)
                    
                    # Reproducir la alarma si aún no se está reproduciendo
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                    
                    # Mostrar información en el frame
                    cv2.putText(frame_processed, "Target", (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame_processed, "Person Detected", (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    count += 1
                    
    # Dibujar el polígono en el frame si se ha definido
    if len(pts) >= 4:
        poly_overlay = frame_processed.copy()
        cv2.fillPoly(poly_overlay, np.array([pts]), (0, 255, 0))
        frame_processed = cv2.addWeighted(poly_overlay, 0.1, frame_processed, 0.9, 0)
    
    cv2.imshow("Video", frame_processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
