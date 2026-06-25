from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ======================
# FLASK
# ======================

app = Flask(__name__)

# ======================
# LOAD MODEL
# ======================

model = load_model("model/sign_model.h5")
labels = np.load("model/labels.npy", allow_pickle=True)

# ======================
# MEDIAPIPE
# ======================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ======================
# 3. CAMERA
# ======================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Kamera gagal dibuka")
else:
    print("✅ Kamera berhasil dibuka")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

hasil_label = "Belum Terdeteksi"
hasil_confidence = 0

# ======================
# VIDEO STREAM
# ======================

def generate_frames():

    global hasil_label
    global hasil_confidence

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Frame gagal dibaca")
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        # 4. mediapipe mendeteksi tangan
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                data = []
                # 5. Mengambil 21 Titik Landmark
                for lm in hand.landmark:

                    data.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])
                # 6. Mode Deep learning melakukan klasifikasi
                data = np.array(data).reshape(1, -1)

                pred = model.predict(
                    data,
                    verbose=0
                )

                # 7. Mengambil Hasil Prediksi
                class_id = np.argmax(pred)

                confidence = float(
                    pred[0][class_id]
                )

                # Mengubah menjadi label gesture
                hasil_label = str(
                    labels[class_id]
                )

                # Mengitung confidence
                hasil_confidence = round(
                    confidence * 100,
                    2
                )

                cv2.putText(
                    frame,
                    f"{hasil_label} ({hasil_confidence}%)",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        else:

            hasil_label = "Tidak Ada Tangan"
            hasil_confidence = 0

            # Menampilkan Hasil pada video
            cv2.putText(
                frame,
                "Tidak Ada Tangan",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# ======================
# ROUTES
# ======================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/predict')
def predict():

    return jsonify({
        "label": hasil_label,
        "confidence": hasil_confidence
    })

@app.route('/test_camera')
def test_camera():

    success, frame = camera.read()

    if success:
        return "✅ Kamera Berjalan"

    return "❌ Kamera Tidak Berjalan"

# ======================
# RUN
# ======================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )