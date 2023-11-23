import joblib
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# Load the trained Decision Tree model
decision_tree = joblib.load("decision_tree_model.joblib")

# Mapping between class index and class label
class_mapping = {0: "bukan_nanas", 1: "nanas_matang", 2: "nanas_mentah"}


# Function to preprocess a single image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_flat = img_array.reshape(1, -1)  # Flatten the image for the Decision Tree
    return img_flat


# Example usage: make predictions on a new image
new_image_path = "testing/nanasmentah1.jpeg"
preprocessed_image = preprocess_image(new_image_path)
predicted_class_index = decision_tree.predict(preprocessed_image)[0]

# Find the class name based on the index
predicted_class_name = class_mapping.get(predicted_class_index, "Unknown")

print(f"Predicted Class: {predicted_class_name}")
