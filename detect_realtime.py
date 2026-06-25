import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# =====================
# LOAD MODEL
# =====================
# Memuat model hasil training 
model = load_model("model/sign_model.h5")
labels = np.load("model/labels.npy", allow_pickle=True)

# =====================
# MEDIAPIPE
# =====================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================
# CAMERA
# =====================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi")
    exit()

print("===================================")
print("Deteksi Bahasa Isyarat Aktif")
print("Tekan Q untuk keluar")
print("===================================")

while True:
    # 1. WebCam menangkap frame
    ret, frame = cap.read()

    if not ret:
        print("❌ Gagal membaca frame kamera")
        continue

    frame = cv2.flip(frame, 1)
    # 2. mediapipe mendeteksi tangan
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    label_text = "Tidak Ada Tangan"
    confidence_text = ""

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            # 3. Menghasilkan 21 landamrk dan mengubahnya menjadi fitur
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            data = np.array(data).reshape(1, -1)

            try:
                # 4. Fitur dikirim ke model untuk diprediksi
                prediction = model.predict(
                    data,
                    verbose=0
                )
                # 5. Mengambil probabilitas tertinggi
                class_id = np.argmax(prediction)

                confidence = float(
                    prediction[0][class_id]
                )
                # 6. lalu kinversi kembali menjadi nama gesture mengunakan label y sudh digunakan sebelumnya 
                label_text = str(
                    labels[class_id]
                )
                # 7. Mengambil nilai confidence 
                confidence_text = (
                    f"{confidence*100:.2f}%"
                )

                color = (
                    (0,255,0)
                    if confidence > 0.80
                    else (0,0,255)
                )

                cv2.putText(
                    frame,
                    label_text,
                    (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )
                # Menampilkan nama gesture dan nilai confidence secara realtime menggunakan OpenCv
                cv2.putText(
                    frame,
                    confidence_text,
                    (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            except Exception as e:

                print(
                    "Error Prediksi :",
                    e
                )

    cv2.imshow(
        "Deteksi Bahasa Isyarat",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()