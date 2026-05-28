import os
import numpy as np
import joblib

from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Folder dataset
dataset_path = "dataset"

labels = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

X = []
y = []

IMG_SIZE = 100

# Membaca dataset
for label in labels:

    folder_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder_path):

        try:

            img_path = os.path.join(folder_path, img_name)

            # buka gambar
            img = Image.open(img_path)

            # ubah grayscale
            img = img.convert("L")

            # resize
            img = img.resize((IMG_SIZE, IMG_SIZE))

            # ubah ke array
            img = np.array(img)

            # flatten
            X.append(img.flatten())

            y.append(label)

        except:
            pass

# Convert array
X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Akurasi
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi Model:", accuracy)

# Simpan model
joblib.dump(model, "model.pkl")

print("Model berhasil disimpan!")