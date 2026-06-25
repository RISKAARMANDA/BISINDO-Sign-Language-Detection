import os
import cv2
import numpy as np
import mediapipe as mp

# =========================
# PATH DATASET
# =========================
# Program membaca seluruh gambar pada folder dataset
DATASET_PATH = r"D:\TI 3A\SEMESTER 6\Pratikum Computer Vision\Project_Bahasa_Isyarat\dataset"

# =========================
# OUTPUT CSV
# =========================
SAVE_PATH = r"csv"

os.makedirs(SAVE_PATH, exist_ok=True)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True, # Untuk memproses gambar static
    max_num_hands=1, # Mendetksi satu tangan
    min_detection_confidence=0.5 # memastikan deteksi minimal 50% 
)

print("===================================")
print("EXTRACT LANDMARK DIMULAI")
print("===================================")

total_data = 0

# =========================
# LOOP LABEL
# =========================

for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\nMemproses Label : {label}")

    data_list = []

    # =========================
    # LOOP GAMBAR
    # =========================
    # 2 Membaa seluruh gmbar yg ad difolder satu persatu
    for file in os.listdir(label_path):

        img_path = os.path.join(label_path, file)

        img = cv2.imread(img_path)

        if img is None:
            print("Gagal baca :", img_path)
            continue
        #3 Mengubah gmabr jadi RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #4 Proses Detekssi Landmark
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                data = []
                #5 Mengambil Kordinat Landmark
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                #6 Menambahkan label simpan ke array
                data.append(label)
                data_list.append(data)

                total_data += 1

    # =========================
    # SAVE CSV
    # =========================
    if len(data_list) > 0:

        data_array = np.array(data_list, dtype=object)

        save_file = os.path.join(SAVE_PATH, f"{label}.csv")
        #8 Menyimpan Menjadi csv
        np.savetxt(
            save_file,
            data_array,
            delimiter=",",
            fmt="%s"
        )

        print(f"CSV disimpan : {save_file}")
        print(f"Jumlah data : {len(data_list)}")

    else:
        print(f"Tidak ada tangan terdeteksi untuk label {label}")

print("\n===================================")
print("SELESAI")
print("Total Data :", total_data)
print("===================================")