import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import img_to_array, load_img # pyright: ignore[reportMissingImports]
from tensorflow.keras import models, layers # pyright: ignore[reportMissingImports]

from sklearn.model_selection import train_test_split


NUM_CATEGORIES = 10
PER_CATEGORY = 1000

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
EPOCHS = 50

DATA_DIR = 'numbers_samples'
IMG_HEIGHT = 128
IMG_WIDTH = 128


def get_model():

    # Create a sequential neural network
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Hidden Convolutional layer (1)
    model.add(layers.Conv2D(64, (3, 3), 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # Hidden Convolutional layer (2)
    model.add(layers.Conv2D(64, (3, 3), 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # Hidden Convolutional layer (3)
    model.add(layers.Conv2D(64, (3, 3),
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.1))
    model.add(layers.Dropout(0.2))

    # Hidden Dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(negative_slope=0.1))
    model.add(layers.Dropout(0.2))

    # Output layer
    model.add(layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model  


def load_data(data_dir):
    """
    Return lists of images and corresponding labels
    """
    images = []
    labels = []

    # Loop through each folder of image classes
    for img_folder in os.listdir(data_dir):

        num_dir = os.path.join(data_dir, img_folder)
        labels.extend([int(img_folder)] * PER_CATEGORY)

        # Loop through each image within each class
        for img_file in os.listdir(num_dir):

            # Load the image as PIL object
            img_path = os.path.join(num_dir, img_file)
            img = load_img(img_path, target_size = (IMG_HEIGHT, IMG_WIDTH))

            # Normalize RGB values
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

    return images, labels

    
def main():

    # Load the data directory
    images, labels = load_data(DATA_DIR)

    # Convert classes to categorical labels
    labels = tf.keras.utils.to_categorical(labels)

    # Split our data into training/validation and testing sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        np.array(images), np.array(labels),  
        test_size = TEST_SIZE, stratify = np.array(labels)
    )

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, 
        test_size = VALIDATION_SIZE, stratify = y_train_val
    )

    # Build the model
    model = get_model()

    # Define EarlyStopping callback
    early_stopping = EarlyStopping( 
        monitor = 'val_loss',
        patience = 3, 
        restore_best_weights = True
    )

    # Fit model on training data
    model.fit(
        x_train, y_train, 
        epochs = EPOCHS, shuffle = True,
        validation_data = (x_val, y_val),
        callbacks = [early_stopping]
    )

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)


main()

