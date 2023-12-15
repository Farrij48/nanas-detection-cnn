# from flask import Flask, request, jsonify
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os
# import io
# import cv2
# from flask_cors import CORS
# import serial
# import time

# app = Flask(__name__)
# CORS(app)

# arduino_port = "COM13"
# ser = serial.Serial(arduino_port, 9600)

# modelh5 = load_model("model7.h5")


# def send_serial_instruction(instruction):
#     ser.write(instruction.encode())


# def predict_model_keras(image_data):
#     # Lakukan prediksi menggunakan model
#     img = image.load_img(io.BytesIO(image_data), target_size=(100, 100))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     hasil = modelh5.predict(img_array)
#     classes = ["bukan_nanas", "nanas_matang", "nanas_mentah", "tidak_ada_object"]
#     prediction = classes[np.argmax(hasil)]

#     if prediction == "nanas_matang":
#         send_serial_instruction("C")
#     elif prediction == "nanas_mentah":
#         send_serial_instruction("M")
#     elif prediction == "bukan_nanas":
#         send_serial_instruction("O")
#     else:
#         send_serial_instruction("C")

#     return jsonify({"prediction": prediction})


# @app.route("/predict", methods=["POST"])
# def predict_image():
#     # Ambil gambar dari request
#     image_file = request.files["image"]

#     if not image_file:
#         return jsonify({"error": "No image file provided"}), 400

#     try:
#         image_data = image_file.read()
#         return predict_model_keras(image_data)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

# from flask import Flask, request, jsonify
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os
# import io
# import cv2
# from flask_cors import CORS
# import serial
# import time

# app = Flask(__name__)
# CORS(app)

# arduino_port = "COM13"
# ser = serial.Serial(arduino_port, 9600)

# modelh5 = load_model("model8.h5")

# last_prediction_time = 0
# time_threshold = 2  # Ambil contoh waktu threshold, sesuaikan dengan kebutuhan Anda


# def send_serial_instruction(instruction):
#     ser.write(instruction.encode())


# def predict_model_keras(image_data):
#     global last_prediction_time

#     # Lakukan prediksi menggunakan model
#     img = image.load_img(io.BytesIO(image_data), target_size=(100, 100))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     hasil = modelh5.predict(img_array)
#     classes = ["bukan_nanas", "nanas_matang", "nanas_mentah", "tidak_ada_object"]
#     prediction = classes[np.argmax(hasil)]

#     current_time = time.time()

#     # Cek apakah waktu antara prediksi sekarang dan prediksi sebelumnya melewati batas threshold
#     if current_time - last_prediction_time >= time_threshold:
#         if prediction == "nanas_matang":
#             send_serial_instruction("C")
#         elif prediction == "nanas_mentah":
#             send_serial_instruction("M")
#         elif prediction == "bukan_nanas":
#             send_serial_instruction("O")
#         else:
#             send_serial_instruction("C")

#         # Perbarui waktu terakhir prediksi
#         last_prediction_time = current_time

#     return jsonify({"prediction": prediction})


# @app.route("/predict", methods=["POST"])
# def predict_image():
#     # Ambil gambar dari request
#     image_file = request.files["image"]

#     if not image_file:
#         return jsonify({"error": "No image file provided"}), 400

#     try:
#         image_data = image_file.read()
#         return predict_model_keras(image_data)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

# from flask import Flask, request, jsonify
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os
# import io
# import cv2
# from flask_cors import CORS
# import serial
# import time

# app = Flask(__name__)
# CORS(app)

# arduino_port = "COM13"
# ser = serial.Serial(arduino_port, 9600)

# modelh5 = load_model("model8.h5")

# last_prediction = None
# last_prediction_time = 0
# time_threshold = 1  # Ambil contoh waktu threshold, sesuaikan dengan kebutuhan Anda


# def send_serial_instruction(instruction):
#     ser.write(instruction.encode())


# def predict_model_keras(image_data):
#     global last_prediction, last_prediction_time

#     # Lakukan prediksi menggunakan model
#     img = image.load_img(io.BytesIO(image_data), target_size=(100, 100))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     hasil = modelh5.predict(img_array)
#     classes = ["bukan_nanas", "nanas_matang", "nanas_mentah", "tidak_ada_object"]
#     current_prediction = classes[np.argmax(hasil)]

#     current_time = time.time()

#     # Cek apakah waktu antara prediksi sekarang dan prediksi sebelumnya melewati batas threshold
#     if current_time - last_prediction_time >= time_threshold:
#         if current_prediction != last_prediction:
#             if current_prediction == "nanas_matang":
#                 send_serial_instruction("C")
#             elif current_prediction == "nanas_mentah":
#                 send_serial_instruction("C")
#                 send_serial_instruction("M")
#             elif current_prediction == "bukan_nanas":
#                 send_serial_instruction("O")
#             else:
#                 send_serial_instruction("C")

#             # Perbarui prediksi dan waktu terakhir prediksi
#             last_prediction = current_prediction
#             last_prediction_time = current_time

#     return jsonify({"prediction": current_prediction})


# @app.route("/predict", methods=["POST"])
# def predict_image():
#     # Ambil gambar dari request
#     image_file = request.files["image"]

#     if not image_file:
#         return jsonify({"error": "No image file provided"}), 400

#     try:
#         image_data = image_file.read()
#         return predict_model_keras(image_data)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import io
import cv2
from flask_cors import CORS
import serial
import time

app = Flask(__name__)
CORS(app)

arduino_port = "COM13"
ser = serial.Serial(arduino_port, 9600)

modelh5_1 = load_model("model9.h5")
modelh5_2 = load_model("model5.h5")

last_prediction = None
last_prediction_time = 0
first_prediction = None  # Tambahkan variabel untuk menyimpan prediksi pertama
time_threshold = 1  # Ambil contoh waktu threshold, sesuaikan dengan kebutuhan Anda


def send_serial_instruction(instruction):
    ser.write(instruction.encode()) 


def predict_model_keras(image_data):
    global last_prediction, last_prediction_time, first_prediction

    # Lakukan prediksi menggunakan model
    img = image.load_img(io.BytesIO(image_data), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    hasil = modelh5_1.predict(img_array)
    classes = ["bukan_nanas", "nanas_matang", "nanas_mentah", "tidak_ada_object"]
    current_prediction = classes[np.argmax(hasil)]

    current_time = time.time()

    # Jika prediksi pertama belum diset, simpan sebagai prediksi pertama
    if first_prediction is None:
        first_prediction = current_prediction

    # Cek apakah waktu antara prediksi sekarang dan prediksi sebelumnya melewati batas threshold
    if current_time - last_prediction_time >= time_threshold:
        if current_prediction != last_prediction:
            if current_prediction == "nanas_matang":
                send_serial_instruction("C")
            elif current_prediction == "nanas_mentah":
                send_serial_instruction("C")
                send_serial_instruction("M")
            elif current_prediction == "bukan_nanas":
                send_serial_instruction("O")
            else:
                send_serial_instruction("C")

            # Perbarui prediksi dan waktu terakhir prediksi
            last_prediction = current_prediction
            last_prediction_time = current_time

    return jsonify({"prediction": current_prediction})


def predict_model_keras2(image_data):
    global last_prediction, last_prediction_time, first_prediction

    # Lakukan prediksi menggunakan model
    img = image.load_img(io.BytesIO(image_data), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    hasil = modelh5_2.predict(img_array)
    classes = ["bukan_nanas", "nanas_matang", "nanas_mentah"]
    current_prediction = classes[np.argmax(hasil)]

    return jsonify({"prediction": current_prediction})


@app.route("/predict", methods=["POST"])
def predict_image():
    # Ambil gambar dari request
    image_file = request.files["image"]

    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_data = image_file.read()
        return predict_model_keras(image_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict2", methods=["POST"])
def predict_image2():
    # Ambil gambar dari request
    image_file = request.files["image"]

    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_data = image_file.read()
        return predict_model_keras2(image_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
