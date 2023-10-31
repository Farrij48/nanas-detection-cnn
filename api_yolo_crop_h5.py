from ultralytics import YOLO
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import io
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model saat aplikasi Flask dimulai
modelh5 = load_model("model2.h5")

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please check the path."
    )

modelyolo = YOLO(model_path)  # load a custom model

# Ambang batas deteksi
threshold_live = 0.5


def predict_model_yolo(image_data):
    image_cv2 = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    results = modelyolo(image_cv2)[0]

    # Proses hasil deteksi
    image_cv2 = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    results = modelyolo(image_cv2)[0]

    # Proses hasil deteksi
    detected_objects = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold_live:
            detected_objects.append(
                {
                    "class": results.names[int(class_id)].upper(),
                    "confidence": score,
                    "bounding_box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    },
                }
            )

    if not detected_objects:
        return jsonify({"prediction": "tidak_diketahui"})
    else:
        # crop gambar yang terdeteksi kemudian lakukan prediksi menggunakan model keras
        x1 = detected_objects[0]["bounding_box"]["x1"]
        y1 = detected_objects[0]["bounding_box"]["y1"]
        x2 = detected_objects[0]["bounding_box"]["x2"]
        y2 = detected_objects[0]["bounding_box"]["y2"]
        image_cv2 = image_cv2[y1:y2, x1:x2]
        image_data = cv2.imencode(".jpg", image_cv2)[1].tostring()
        return predict_model_keras(image_data)


def predict_model_keras(image_data):
    # Lakukan prediksi menggunakan model
    img = image.load_img(io.BytesIO(image_data), target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    hasil = modelh5.predict(img_array)
    if hasil[0][0] > hasil[0][1]:
        prediction = "nanas_matang"
    else:
        prediction = "nanas_mentah"

    # tampilkan gambar
    # cv2.imshow("image", img_array[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return jsonify({"prediction": prediction})


@app.route("/predict", methods=["POST"])
def predict_image():
    # Ambil gambar dari request
    image_file = request.files["image"]

    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_data = image_file.read()
        return predict_model_yolo(image_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
