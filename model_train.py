import tensorflow as tf
import os
import random
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define image size and batch size
IMAGE_SIZE = (128, 128)  # You can experiment with different image sizes
BATCH_SIZE = 32

# Path to the dataset (replace with your actual path)
train_dir = r"D:\CO542\project\Neuralnetworkproject_chala\Neuralnetworkproject_chala\train"

# Data Augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=30,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Randomly shear images
    zoom_range=0.2,  # Randomly zoom images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill empty pixels after transformations
)

# Load images and their labels into separate lists
class_names = os.listdir(train_dir)
image_paths = []
labels = []

# Collect image paths and their respective labels
for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(class_name)

# Shuffle the image paths and labels together
combined = list(zip(image_paths, labels))
random.shuffle(combined)

# Separate shuffled data back into image paths and labels
image_paths, labels = zip(*combined)

# Custom Data Generator
def custom_data_generator(image_paths, labels, batch_size, class_names):
    num_classes = len(class_names)
    while True:  # Infinite loop for training
        images = []
        batch_labels = []
        batch_image_names = []  # Store image filenames for printing

        for i in range(len(image_paths)):
            img_path = image_paths[i]
            label = labels[i]

            # Load image and convert it to array
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0  # Normalize image
            images.append(img_array)
            batch_labels.append(label)
            batch_image_names.append(img_path)

            # When batch is full, yield the batch and shuffle the data again
            if len(images) == batch_size:
                # Convert labels to categorical (one-hot encoding)
                label_indices = [class_names.index(label) for label in batch_labels]
                label_one_hot = tf.keras.utils.to_categorical(label_indices, num_classes=num_classes)

                # Print image name and its label
                for i in range(batch_size):
                    print(f"Image: {batch_image_names[i]} | Label: {batch_labels[i]}")

                # Yield the batch
                yield np.array(images), label_one_hot
                images, batch_labels, batch_image_names = [], [], []  # Reset batch

        # For remaining images that didn't fill a full batch
        if len(images) > 0:
            label_indices = [class_names.index(label) for label in batch_labels]
            label_one_hot = tf.keras.utils.to_categorical(label_indices, num_classes=num_classes)

            for i in range(len(images)):
                print(f"Image: {batch_image_names[i]} | Label: {batch_labels[i]}")

            yield np.array(images), label_one_hot

# Build the CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolution layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Dropout to prevent overfitting
model.add(Dropout(0.5))

# Output layer (3 classes for dry, oily, normal)
model.add(Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with custom data generator
history = model.fit(
    custom_data_generator(image_paths, labels, BATCH_SIZE, class_names),
    steps_per_epoch=len(image_paths) // BATCH_SIZE,
    epochs=20,
    verbose=2
)

# Save the trained model
model.save("skin_type_model.h5")
