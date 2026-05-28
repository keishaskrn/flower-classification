from flask import Flask, render_template, request
import joblib
import numpy as np
import os

from PIL import Image

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # buka gambar
    img = Image.open(filepath)

    # grayscale
    img = img.convert("L")

    # resize
    img = img.resize((100,100))

    # ubah array
    img = np.array(img)

    # flatten
    img = img.flatten().reshape(1, -1)

    # prediksi
    prediction = model.predict(img)

    hasil = prediction[0]

    return render_template(
        "index.html",
        prediction=hasil,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)