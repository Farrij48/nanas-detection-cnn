from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

model = load_model("model.h5")


def predict_image(filename):
    img = image.load_img(filename, target_size=(100, 100))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    result = model.predict(img)
    if result[0][0] > result[0][1]:
        print("nanas_matang")
    else:
        print("nanas_mentah")
    plt.imshow(img[0])
    plt.show()


predict_image(os.path.join("testing", "nanasmatang1.jpeg"))
predict_image(os.path.join("testing", "nanasmatang2.jpeg"))
predict_image(os.path.join("testing", "nanasmatang3.jpeg"))
predict_image(os.path.join("testing", "nanasmatang4.jpeg"))
predict_image(os.path.join("testing", "nanasmentah1.jpeg"))
predict_image(os.path.join("testing", "nanasmentah2.jpeg"))
predict_image(os.path.join("testing", "nanasmentah3.jpeg"))
predict_image(os.path.join("testing", "nanasmentah4.jpeg"))
predict_image(os.path.join("testing", "pisang1.jpeg"))
predict_image(os.path.join("testing", "tomat1.jpeg"))
