import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


base_dir = "dataset"
train_path = os.path.join(base_dir, "train")
test_path = os.path.join(base_dir, "validation")

model = Sequential()
model.add(Conv2D(128, (3, 3), activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(5000, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()


model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(lr=0.0001),
    metrics=["accuracy"],
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(100, 100),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
)

total_train_samples = len(train_generator.filenames)
total_validation_samples = len(test_generator.filenames)

model.fit(
    train_generator,
    steps_per_epoch=total_train_samples // batch_size,
    epochs=100,
    validation_data=test_generator,
    validation_steps=total_validation_samples // batch_size,
)

model.save("model.h5")

# model.save_weights("deneme.h5")
# plt.figure(figsize=(6, 4))
# plt.plot(history.history["loss"], label="loss")
# plt.plot(history.history["val_loss"], label="validation_loss")
# plt.legend()
# plt.show()
# plt.figure(figsize=(6, 4))
# plt.plot(history.history["accuracy"], label="accuracy")
# plt.plot(history.history["val_accuracy"], label="validation accuracy")
# plt.legend()
# plt.show()
