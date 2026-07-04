import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename

# ==============================
# FLASK CONFIG
# ==============================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/coffee_model.keras"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================

model = load_model(MODEL_PATH)

# Label harus sama dengan folder dataset
class_names = [
    "Dark",
    "Green",
    "Light",
    "Medium"
]

# ==============================
# FUNGSI PREDIKSI
# ==============================

def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    confidence = float(np.max(prediction))

    class_index = np.argmax(prediction)

    label = class_names[class_index]

    return label, confidence

# ==============================
# HOME
# ==============================

@app.route("/")

def index():

    return render_template("index.html")

# ==============================
# PREDIKSI
# ==============================

@app.route("/predict", methods=["POST"])

def predict():

    if "image" not in request.files:
        return "Tidak ada gambar."

    file = request.files["image"]

    if file.filename == "":
        return "Pilih gambar terlebih dahulu."

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    label, confidence = predict_image(filepath)

    return render_template(
        "result.html",
        image=filename,
        label=label,
        confidence=round(confidence * 100, 2)
    )

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)