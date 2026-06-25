import cv2
import os
import mediapipe as mp

# Menentukan Label dataset
label = "Mulai"

# Folder penyimpanan
save_path = os.path.join("dataset", label)
os.makedirs(save_path, exist_ok=True)

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1, # Membatasi deteksi hanya satu tangan 
    min_detection_confidence=0.7 # Memastikan deteksi dilakukan dengan tidak kepercayaan minimal 70 %
)

# Mengaktifkan Kamera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera tidak terdeteksi")
    exit()

count = len(os.listdir(save_path))

print("S = Simpan Data")
print("Q = Keluar")

# Proses dilakukan berulang kali sampaijumlah data mencukupi
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb) # Memproses gambar untuk mendeteksi tngan.

    # Menampilkan landmark dalm bntuk titik dan garis
    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    # Informasi
    cv2.putText(
        frame,
        f"Label : {label}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Data : {count}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Tampilkan kamera
    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # Menyimpan gambar 
    if key == ord('s'):

        filename = os.path.join(
            save_path,
            f"{count}.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved :", filename)

        count += 1

    # Keluar
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()