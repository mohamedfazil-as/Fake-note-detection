import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset folders
REAL_FOLDER = 'dataset/real'
FAKE_FOLDER = 'dataset/fake'

# Ensure dataset directories exist
if not os.path.exists(REAL_FOLDER) or not os.path.exists(FAKE_FOLDER):
    raise FileNotFoundError("❌ Dataset folders not found. Ensure 'dataset/real' and 'dataset/fake' exist.")

# Load images and labels
def load_images_and_labels():
    images, labels = [], []
    real_count, fake_count = 0, 0  # Debugging counters

    for filename in os.listdir(REAL_FOLDER):
        img_path = os.path.join(REAL_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(1)  # ✅ Real Note
        real_count += 1

    for filename in os.listdir(FAKE_FOLDER):
        img_path = os.path.join(FAKE_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(0)  # ✅ Fake Note
        fake_count += 1

    print(f"🔍 Total Real Notes: {real_count}")
    print(f"🔍 Total Fake Notes: {fake_count}")

    if real_count == 0 or fake_count == 0:
        raise ValueError("❌ Dataset is imbalanced! Add more images.")

    images = np.array(images).astype(np.float32) / 255.0  # Normalize
    labels = np.array(labels).reshape(-1, 1)  # Correct shape
    return images, labels

# Create CNN Model
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    images, labels = load_images_and_labels()

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    
    model = create_model()

    # Train model
    model.fit(datagen.flow(images, labels, batch_size=32, shuffle=True), epochs=10)

    # Save the model
    os.makedirs('model', exist_ok=True)
    model.save('model/fake_note_detector.h5')
    print("✅ Model training complete and saved to model/fake_note_detector.h5")
