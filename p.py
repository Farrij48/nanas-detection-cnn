import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d
from keras.preprocessing.image import load_img, img_to_array

# Define dataset paths
base_dir = "dataset7"
train_path = os.path.join(base_dir, "train")
test_path = os.path.join(base_dir, "validation")


# Load and preprocess data
def load_and_preprocess_data(directory):
    X = []
    y = []
    label_encoder = LabelEncoder()

    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                img = load_img(image_path, target_size=(150, 150))
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label)

    X = np.array(X)
    y = label_encoder.fit_transform(y)

    return X, y


# Load and preprocess training data
X_train, y_train = load_and_preprocess_data(train_path)

# Load and preprocess validation data
X_val, y_val = load_and_preprocess_data(test_path)

# Flatten images for Decision Tree input
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Train Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_flat, y_train)

# Make predictions on the validation set
y_val_pred = decision_tree.predict(X_val_flat)

# Evaluate accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy}")

# Save the trained Decision Tree model
import joblib

model_filename = "decision_tree_model.joblib"
joblib.dump(decision_tree, model_filename)
print(f"Model saved as {model_filename}")
