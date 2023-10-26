from ultralytics import YOLO
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import io
import cv2
from flask_cors import CORS

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, regionprops_table
from collections import Counter

app = Flask(__name__)
CORS(app)

# Load model saat aplikasi Flask dimulai
modelh5 = load_model("model.h5")

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please check the path."
    )

modelyolo = YOLO(model_path)  # load a custom model

# Ambang batas deteksi
threshold_live = 0.5


# knn function
def knn_predict(file_name):
    file = "features.xlsx"
    file_name = file_name
    dataset = pd.read_excel(file)
    glcm_properties = [
        "dissimilarity",
        "correlation",
        "homogeneity",
        "contrast",
        "ASM",
        "energy",
    ]

    fitur = dataset.iloc[:, +1:-1].values
    kelas = dataset.iloc[:, 30].values
    tes_fitur = []
    tes_fitur.append([])
    tes_kelas = []
    tes_kelas.append([])
    tes_kelas[0].append("nanas_matang")
    # print(len(fitur))
    # Feature extraction for data testing----------------------------------------------
    # Preprocessing
    src = cv2.imread(file_name, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask.copy(), None, iterations=10)
    mask = cv2.erode(mask.copy(), None, iterations=10)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, mask]
    dst = cv2.merge(rgba, 4)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selected = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(selected)
    cropped = dst[y : y + h, x : x + w]
    mask = mask[y : y + h, x : x + w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # HSV
    hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
    clt = KMeans(n_clusters=3, n_init=10)  # Atur nilai n_init secara eksplisit
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    tes_fitur[0].append(dom_color[0])
    tes_fitur[0].append(dom_color[1])
    tes_fitur[0].append(dom_color[2])
    # GLCM
    glcm = graycomatrix(
        gray,
        distances=[5],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    feature = []
    glcm_props = [
        propery for name in glcm_properties for propery in graycoprops(glcm, name)[0]
    ]
    for item in glcm_props:
        tes_fitur[0].append(item)

    # Shape
    label_img = label(mask)
    props = regionprops(label_img)
    eccentricity = getattr(props[0], "eccentricity")
    area = getattr(props[0], "area")
    perimeter = getattr(props[0], "perimeter")
    metric = (4 * np.pi * area) / (perimeter * perimeter)
    tes_fitur[0].append(metric)
    tes_fitur[0].append(eccentricity)
    # --------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(fitur)
    fitur = scaler.transform(fitur)
    tes_fitur = scaler.transform(tes_fitur)

    classifier = KNeighborsClassifier(n_neighbors=13)
    classifier.fit(fitur, kelas)

    kelas_pred = classifier.predict(tes_fitur)
    return kelas_pred[0]


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
        return jsonify({"prediction": "tidak_diketahui"}), 200

    for obj in detected_objects:
        x1 = obj["bounding_box"]["x1"]
        y1 = obj["bounding_box"]["y1"]
        x2 = obj["bounding_box"]["x2"]
        y2 = obj["bounding_box"]["y2"]

        crop_img = image_cv2
        cv2.imwrite("crop.jpg", crop_img)
        hasil_knn = knn_predict("crop.jpg")
        hasil_cnn = predict_model_keras(image_data)

        return (
            jsonify(
                {
                    "prediction": "Hasil Prediksi CNN : "
                    + hasil_cnn
                    + " Hasil Prediksi KNN : "
                    + hasil_knn
                }
            ),
            200,
        )


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

    return prediction


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
