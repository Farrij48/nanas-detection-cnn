# from ultralytics import YOLO
# from flask import Flask, request, jsonify
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os
# import io
# import cv2
# from flask_cors import CORS
# import serial

# app = Flask(__name__)
# CORS(app)


# arduino_port = "COM13"  # Ganti dengan port Arduino yang sesuai
# ser = serial.Serial(arduino_port, 9600)  # Buka koneksi serial dengan baud rate 9600

# # Load model saat aplikasi Flask dimulai
# modelh5 = load_model("model5.h5")

# # Inisialisasi model YOLO dan model path
# model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")

# if not os.path.exists(model_path):
#     raise FileNotFoundError(
#         f"Model file '{model_path}' not found. Please check the path."
#     )

# modelyolo = YOLO(model_path)  # load a custom model


# def send_serial_instruction(instruction):
#     ser.write(instruction.encode())


# def predict_model_yolo(image_data):
#     image_cv2 = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
#     results = modelyolo(image_cv2)[0]

#     # Proses hasil deteksi
#     detected_objects = []
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         detected_objects.append(
#             {
#                 "class": results.names[int(class_id)].upper(),
#                 "confidence": score,
#                 "bounding_box": {
#                     "x1": int(x1),
#                     "y1": int(y1),
#                     "x2": int(x2),
#                     "y2": int(y2),
#                 },
#             }
#         )

#     if not detected_objects:
#         send_serial_instruction("S")  # Instruksi untuk menghentikan conveyor dan servo
#         return jsonify({"prediction": "tidak_diketahui"})
#     else:
#         # crop gambar yang terdeteksi kemudian lakukan prediksi menggunakan model keras
#         x1 = detected_objects[0]["bounding_box"]["x1"]
#         y1 = detected_objects[0]["bounding_box"]["y1"]
#         x2 = detected_objects[0]["bounding_box"]["x2"]
#         y2 = detected_objects[0]["bounding_box"]["y2"]
#         image_cv2 = image_cv2[y1:y2, x1:x2]
#         image_data = cv2.imencode(".jpg", image_cv2)[1].tostring()
#         send_serial_instruction("C")  # Instruksi untuk mengaktifkan conveyor dan servo
#         return predict_model_keras(image_data)


# def predict_model_keras(image_data):
#     # Lakukan prediksi menggunakan model
#     img = image.load_img(io.BytesIO(image_data), target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     hasil = modelh5.predict(img_array)
#     classes = ["bukan_nanas", "nanas_matang", "nanas_mentah"]
#     prediction = classes[np.argmax(hasil)]

#     if prediction == "nanas_matang":
#         send_serial_instruction(
#             "C"
#         )  # Instruksi untuk memilah buah matang menggunakan servo
#     elif prediction == "nanas_mentah":
#         send_serial_instruction(
#             "M"
#         )  # Instruksi untuk memilah buah mentah menggunakan servo
#     else:
#         send_serial_instruction("O")  # Instruksi untuk menghentikan conveyor dan servo

#     return jsonify({"prediction": prediction})


# @app.route("/predict", methods=["POST"])
# def predict_image():
#     # Ambil gambar dari request
#     image_file = request.files["image"]

#     if not image_file:
#         return jsonify({"error": "No image file provided"}), 400

#     try:
#         image_data = image_file.read()
#         return predict_model_yolo(image_data)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

from ultralytics import YOLO
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import io
import cv2
from flask_cors import CORS
import serial
import time  # Import modul time

app = Flask(__name__)
CORS(app)

arduino_port = "COM13"  # Ganti dengan port Arduino yang sesuai
ser = serial.Serial(arduino_port, 9600)  # Buka koneksi serial dengan baud rate 9600

# Load model saat aplikasi Flask dimulai
modelh5 = load_model("model5.h5")

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please check the path."
    )

modelyolo = YOLO(model_path)  # load a custom model


def send_serial_instruction(instruction):
    ser.write(instruction.encode())


def predict_model_yolo(image_data):
    image_cv2 = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    results = modelyolo(image_cv2)[0]

    # Proses hasil deteksi
    detected_objects = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

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
        send_serial_instruction("S")  # Instruksi untuk menghentikan conveyor dan servo
        return jsonify({"prediction": "tidak_diketahui"})
    else:
        # crop gambar yang terdeteksi kemudian lakukan prediksi menggunakan model keras
        x1 = detected_objects[0]["bounding_box"]["x1"]
        y1 = detected_objects[0]["bounding_box"]["y1"]
        x2 = detected_objects[0]["bounding_box"]["x2"]
        y2 = detected_objects[0]["bounding_box"]["y2"]
        image_cv2 = image_cv2[y1:y2, x1:x2]
        image_data = cv2.imencode(".jpg", image_cv2)[1].tostring()
        send_serial_instruction("C")  # Instruksi untuk mengaktifkan conveyor dan servo

        # Tambahkan waktu delay sebelum mengirim frame berikutnya (misalnya, 2 detik)
        delay_time = 5  # sesuaikan dengan kebutuhan Anda
        time.sleep(delay_time)

        return predict_model_keras(image_data)


def predict_model_keras(image_data):
    # Lakukan prediksi menggunakan model
    img = image.load_img(io.BytesIO(image_data), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    hasil = modelh5.predict(img_array)
    classes = ["bukan_nanas", "nanas_matang", "nanas_mentah"]
    prediction = classes[np.argmax(hasil)]

    if prediction == "nanas_matang":
        send_serial_instruction(
            "C"
        )  # Instruksi untuk memilah buah matang menggunakan servo
    elif prediction == "nanas_mentah":
        send_serial_instruction(
            "M"
        )  # Instruksi untuk memilah buah mentah menggunakan servo
    else:
        send_serial_instruction("O")  # Instruksi untuk menghentikan conveyor dan servo

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
