from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt


model = load_model("model5.h5")


def predict_image(filename):
    img = image.load_img(filename, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    result = model.predict(img)

    classes = ["bukan nanas", "nanas_matang", "nanas_mentah"]
    print(classes[np.argmax(result)])

    # tampilkan gambar tanpa resize
    img = plt.imread(filename)
    plt.imshow(img)
    plt.show()


# Contoh penggunaan
predict_image(os.path.join("testing", "nanasmatang1.jpeg"))
predict_image(os.path.join("testing", "nanasmatang2.jpeg"))
predict_image(os.path.join("testing", "nanasmatang3.jpeg"))
predict_image(os.path.join("testing", "nanasmatang4.jpeg"))
predict_image(os.path.join("testing", "nanasmentah1.jpeg"))
predict_image(os.path.join("testing", "nanasmentah2.jpeg"))
predict_image(os.path.join("testing", "nanasmentah3.jpeg"))
predict_image(os.path.join("testing", "nanasmentah4.jpeg"))

predict_image(os.path.join("dataset7", "validation", "bukan_nanas", "apel114.png"))
predict_image(os.path.join("dataset7", "validation", "bukan_nanas", "apel115.png"))
predict_image(os.path.join("dataset7", "validation", "bukan_nanas", "apel116.png"))
predict_image(os.path.join("dataset7", "validation", "bukan_nanas", "apel117.png"))
predict_image(os.path.join("dataset7", "validation", "bukan_nanas", "apel118.png"))
predict_image(os.path.join("dataset7", "validation", "bukan_nanas", "apel119.png"))
predict_image(os.path.join("dataset7", "validation", "bukan_nanas", "apel120.png"))
