from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # increase if time permits
NUM_CLASSES = 6  # Gir, Holstein, Sahiwal, Murrah, Jaffarabadi, Surti

# Load base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=out)

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory("dataset/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
val_gen = val_datagen.flow_from_directory("dataset/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
model.save("cattle_buffalo_model.h5")

print("✅ Training complete. Model saved as cattle_buffalo_model.h5")
