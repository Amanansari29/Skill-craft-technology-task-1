*dataset download code*                                                                                                                                                               # Step 1: Install the Kaggle API
!pip install kaggle

# Step 2: Upload your kaggle.json file
from google.colab import files
uploaded = files.upload()  # Click "Choose Files" and select your kaggle.json

# Step 3: Set up the Kaggle configuration
import os
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Download your dataset
!kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset

# Step 5: Unzip the downloaded dataset
import zipfile
with zipfile.ZipFile('dog-and-cat-classification-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

# Step 6: Verify the downloaded files
import os
dataset_path = 'dataset'
print("Files in dataset directory:")
for item in os.listdir(dataset_path):
    print(f"  - {item}")

# Optional: If you want to see the structure of subdirectories
print("\nExploring dataset structure:")
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files in each directory
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files)-5} more files')
      *training code*                                                                                                                                                                                              import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# -------------------------------
# 1. Paths & Parameters
# -------------------------------
dataset_path = "/content/dataset/PetImages"
img_size = (224, 224)
batch_size = 32

# -------------------------------
# 2. Data Generators
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class indices:", train_generator.class_indices)
# Should show: {'Cat': 0, 'Dog': 1}

# -------------------------------
# 3. Model (Transfer Learning with MobileNetV2)
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze base initially

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# -------------------------------
# 4. Train
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# -------------------------------
# 5. Fine-tune (optional)
# -------------------------------
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# -------------------------------
# 6. Prediction on Custom Image
# -------------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"

    plt.imshow(image.load_img(img_path))
    plt.title(f"Prediction: {label} ({pred:.4f})")
    plt.axis("off")
    plt.show()

# Example:
predict_image("/content/dataset/PetImages/Cat/1.jpg")
predict_image("/content/dataset/PetImages/Dog/2.jpg")
*load model leter*                                                                                                                                                                                                     from tensorflow.keras.models import load_model
import json

# Load model
model = load_model("cat_dog_model.keras")

# Load class indices (if you saved them)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

print("Class indices:", class_indices)
*prediction code*                                                                                                                                                                                     from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img_path = "/content/dataset/PetImages/Dog/2.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0][0]
label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"

plt.imshow(image.load_img(img_path))
plt.title(f"Prediction: {label} ({pred:.4f})")
plt.axis("off")
plt.show()
