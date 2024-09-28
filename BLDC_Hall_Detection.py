import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16, ResNet50
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
import time
import random

# Specify the directory where your images are stored
image_directory = "E:/USB/논문/논문/S_SCI_BLDC_HALL_SENSOR_AI_IEEE_ACCESS/데이터/0.01/augmented"

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

# Shuffle the list of image files
random.shuffle(image_files)

# Separate filenames and labels for binary classification
labels = [0 if filename.startswith("no_delay") else 1 for filename in image_files]
file_paths = [os.path.join(image_directory, filename) for filename in image_files]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Define the number of classes
num_classes = 2

# Function to load and preprocess image data
def load_and_preprocess_image(file_path, label):
    img = image.load_img(file_path, target_size=(224, 224))
    #img = image.load_img(file_path, target_size=(129, 129))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return img_array, label

# Load and preprocess training data
train_data = [load_and_preprocess_image(file_path, label) for file_path, label in zip(X_train, y_train)]
X_train_processed, y_train_processed = zip(*train_data)
X_train_processed = np.array(X_train_processed)
y_train_processed = to_categorical(y_train_processed, num_classes=num_classes)

# Load and preprocess testing data
test_data = [load_and_preprocess_image(file_path, label) for file_path, label in zip(X_test, y_test)]
X_test_processed, y_test_processed = zip(*test_data)
X_test_processed = np.array(X_test_processed)
y_test_processed = to_categorical(y_test_processed, num_classes=num_classes)

# Function to create and train a MobileNet model
def create_and_train_model(model, X_train, y_train, X_test, y_test, model_name):
    base_model = model(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Adjust the input shape based on the specific model requirements

    # Add your own top layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    training_time = time.time() - start_time

    # Evaluate the model
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)

    # Convert predictions to labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Print classification report
    print(f'{model_name} Classification Report:')
    print(classification_report(y_test_labels, y_pred_labels, target_names=[str(i) for i in range(num_classes)]))

    # Display additional information
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'{model_name} Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
    print(f'{model_name} Parameters: {model.count_params()}')
    print(f'{model_name} Training Time: {training_time:.2f} seconds')
    print(f'{model_name} Inference Time per Sample: {inference_time:.6f} seconds')

# Compare MobileNetV1 and MobileNetV2
create_and_train_model(MobileNet, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'MobileNetV1')
create_and_train_model(MobileNetV2, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'MobileNetV2')
create_and_train_model(VGG16, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'VGG16')
create_and_train_model(ResNet50, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'ResNet50')
create_and_train_model(DenseNet121, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'DenseNet121')
create_and_train_model(InceptionV3, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'InceptionV3')
create_and_train_model(Xception, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'Xception')
create_and_train_model(EfficientNetB0, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'EfficientNetB0')