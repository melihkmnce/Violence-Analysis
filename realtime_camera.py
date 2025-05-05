import cv2
import numpy as np
import time
from threading import Thread
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
import pygame
import os

# =============== THREAD TABANLI KAMERA SINIFI ===============
class VideoStream:
    def __init__(self, src):
        self.src = src
        self.stream = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream.isOpened():
                self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# =============== MODEL ve AYARLAR ===============
model = load_model("violence_detection_model.h5")
font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 32)

prediction_history = []
window_size = 15
prediction_interval = 0.5
last_prediction_time = 0
avg_pred = 0
alarm_played = False

pygame.mixer.init()

# Video kayıt ayarları
recording = False
out = None

# =============== KAMERA BAĞLANTISI YAPISI ===============
def connect_camera(ip_url):
    print("[INFO] Kameraya bağlanılıyor...")
    try:
        stream = VideoStream(ip_url).start()
        time.sleep(2)
        test_frame = stream.read()
        if test_frame is None or test_frame.size == 0:
            raise Exception("Görüntü alınamıyor.")
        print("[INFO] Kamera bağlantısı başarılı.")
        return stream
    except Exception as e:
        print(f"[HATA] Bağlantı kurulamadı: {e}")
        return None

ip = "192.168.1.27"
camera_url = f"http://{ip}:8080/video"
vs = connect_camera(camera_url)
missing_frame_count = 0

# =============== ANA DÖNGÜ ===============
while True:
    if vs is None:
        print("[Uyarı] Tekrar bağlantı deneniyor...")
        time.sleep(5)
        vs = connect_camera(camera_url)
        continue

    frame = vs.read()
    if frame is None:
        missing_frame_count += 1
        print(f"[Uyarı] Kare alınamadı. ({missing_frame_count}/5)")
        time.sleep(1)

        if missing_frame_count >= 5:
            print("[Uyarı] Kare alınamadı. Bağlantı sıfırlanıyor.")
            vs.stop()
            vs = None
            missing_frame_count = 0
        continue
    else:
        missing_frame_count = 0

    current_time = time.time()
    if current_time - last_prediction_time >= prediction_interval:
        last_prediction_time = current_time

        resized = cv2.resize(frame, (128, 128))
        img_array = img_to_array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]
        prediction_history.append(prediction)
        if len(prediction_history) > window_size:
            prediction_history.pop(0)

        avg_pred = np.mean(prediction_history)

    # Yazıyı belirle
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if avg_pred > 0.7:
        text = f"ŞİDDET TESPİT EDİLDİ (%{avg_pred * 100:.1f})"
        color = (255, 0, 0)

        if not alarm_played:
            pygame.mixer.music.load("alarm.wav")
            pygame.mixer.music.play()
            alarm_played = True

            with open("log.txt", "a") as log:
                log.write(f"[{datetime.now()}] Şiddet tespit edildi. Tahmin: %{avg_pred * 100:.1f}\n")

            os.makedirs("videos", exist_ok=True)
            date_folder = datetime.now().strftime("%d-%m-%Y")
            folder_path = os.path.join("videos", date_folder)
            os.makedirs(folder_path, exist_ok=True)
            filename = os.path.join(folder_path, f"violence_{datetime.now().strftime('%H-%M-%S')}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True

    elif avg_pred > 0.4:
        text = f"ŞÜPHELİ HAREKET (%{avg_pred * 100:.1f})"
        color = (255, 255, 0)
        alarm_played = False
        if recording:
            out.release()
            recording = False
    else:
        text = f"Şiddet Yok (%{(1 - avg_pred) * 100:.1f})"
        color = (0, 255, 0)
        alarm_played = False
        if recording:
            out.release()
            recording = False

    # Türkçe yazı ekle
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 30), f"{text} - {timestamp}", font=font, fill=color)
    frame_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Video kaydı varsa yaz
    if recording:
        out.write(frame)

    # Görüntü göster
    cv2.imshow("Şiddet Tespiti (Canlı)", cv2.resize(frame_with_text, (960, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if vs:
    vs.stop()
if recording and out:
    out.release()
cv2.destroyAllWindows()
