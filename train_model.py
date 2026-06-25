import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# PATH CSV
# =========================
CSV_PATH = r"csv"

# =========================
# LOAD DATA
# =========================
X = []
y = []

print("===================================")
print("MEMBACA DATASET CSV")
print("===================================")

#1 Membaca Dataset
for file in os.listdir(CSV_PATH):

    # HANYA FILE CSV
    if file.endswith(".csv"):

        # SKIP dataset.csv
        if file == "dataset.csv":
            print("Skip :", file)
            continue

        file_path = os.path.join(CSV_PATH, file)

        print(f"Memproses : {file}")

        try:
            # 2 Membaca isi file csv
            data = np.loadtxt(
                file_path,
                delimiter=",",
                dtype=str
            )

            # JIKA HANYA 1 BARIS
            if len(data.shape) == 1:
                data = np.array([data])

            for row in data:

                try:
                    #3 Menyimpan fitur dan label 
                    features = row[:-1].astype(np.float32)

                    label = row[-1]

                    X.append(features)
                    y.append(label)

                except:
                    continue

        except Exception as e:
            print("ERROR :", e)

# =========================
# UBAH KE ARRAY
# =========================
X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\n===================================")
print("Jumlah Data :", len(X))
print("Jumlah Fitur :", X.shape[1])
print("===================================")

# =========================
# MENGUBAH LABEL MENAJDI NUMERIK
# =========================
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# SAVE LABEL
os.makedirs("model", exist_ok=True)

np.save(
    "model/labels.npy",
    encoder.classes_
)

# ONE HOT
y_categorical = to_categorical(y_encoded)

# =========================
# Membagi Data Training dan Testing
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

print("\nTrain :", len(X_train))
print("Test  :", len(X_test))

# =========================
# MODEL
# =========================
model = Sequential([

    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(y_categorical.shape[1], activation='softmax')

])

# =========================
# TRAINING MODEL
# =========================
model.compile(
    optimizer='adam',
    
    #Evaluasi Akurasi
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# =========================
# PROSES TRAINING
# =========================
print("\n===================================")
print("TRAINING DIMULAI")
print("===================================")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

# =========================
# EVALUASI
# =========================
loss, accuracy = model.evaluate(X_test, y_test)

print("\n===================================")
print(f"Akurasi : {accuracy*100:.2f}%")
print("===================================")

# =========================
# SAVE MODEL
# =========================
model.save("model/sign_model.h5")

print("\n===================================")
print("MODEL BERHASIL DISIMPAN")
print("Lokasi : model/sign_model.h5")
print("===================================")