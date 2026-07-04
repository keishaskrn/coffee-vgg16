# Placeholder: use the full training code provided in chat.
print('Add full train_model.py here')
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# KONFIGURASI
# =====================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 20

TRAIN_DIR = "dataset/train"

VALIDATION_DIR = "dataset/validation"

TEST_DIR = "dataset/test"

MODEL_DIR = "model"

MODEL_NAME = "coffee_model.keras"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME
)

# =====================================================
# DATA AUGMENTATION
# =====================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=25,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode="nearest"

)

validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical"

)

validation_generator = validation_datagen.flow_from_directory(

    VALIDATION_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical"

)

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

print("=" * 50)

print("Jumlah Kelas :", train_generator.num_classes)

print("Nama Kelas :")

print(train_generator.class_indices)

print("=" * 50)

# =====================================================
# LOAD MODEL VGG16
# =====================================================

base_model = VGG16(

    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3)

)

# Freeze semua layer

for layer in base_model.layers:

    layer.trainable = False

print("Base Model Loaded")

# =====================================================
# MENAMBAHKAN LAYER BARU
# =====================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(

    512,

    activation="relu"

)(x)

x = Dropout(

    0.5

)(x)

x = Dense(

    256,

    activation="relu"

)(x)

x = Dropout(

    0.3

)(x)

predictions = Dense(

    train_generator.num_classes,

    activation="softmax"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=predictions

)

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# =====================================================
# CALLBACK
# =====================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3,

    min_lr=0.000001,

    verbose=1

)

# =====================================================
# TRAINING TAHAP PERTAMA
# =====================================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[

        early_stop,

        checkpoint,

        reduce_lr

    ]

)
# =====================================================
# FINE TUNING
# =====================================================

print("\nMemulai Fine Tuning...\n")

# Buka 4 layer terakhir VGG16
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# =====================================================
# EVALUASI MODEL
# =====================================================

print("\nEvaluasi Model...\n")

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("=" * 50)
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")
print("=" * 50)

# =====================================================
# SIMPAN MODEL
# =====================================================

model.save(MODEL_PATH)

print("\nModel berhasil disimpan!")

print(MODEL_PATH)

# =====================================================
# GRAFIK TRAINING
# =====================================================

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.subplot(1,2,2)

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig("training_result.png")

plt.show()

print("\nTraining selesai.")
print("Grafik disimpan sebagai training_result.png")

# =====================================================
# PREDIKSI CONTOH
# =====================================================

print("\nLabel Kelas:")

for name, idx in train_generator.class_indices.items():
    print(f"{idx} : {name}")

print("\nSelesai.")
