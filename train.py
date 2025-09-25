import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os


data_dir = "Data"
model_dir = "Model"
os.makedirs(model_dir, exist_ok=True)


img_size = 224
batch_size = 32

#Data generator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Saving class labels
labels = list(train_data.class_indices.keys())
with open(os.path.join(model_dir, "labels.txt"), "w") as f:
    for label in labels:
        f.write(f"{label}\n")

# Loading MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,  # exclude final FC layer
    weights="imagenet"  # use pretrained weights
)

base_model.trainable = False  # freeze base model for transfer learning

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save(os.path.join(model_dir, "keras_model.h5"))

print("✅ MobileNetV2 model and labels saved in 'Models/'")
