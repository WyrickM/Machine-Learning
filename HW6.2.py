########################################################################
# Mantz Wyrick
# Machine Learning Homework #6.2
#
# deisgn and compare 3 different CNNs.
#
########################################################################

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Load dataset

# Using MNIST handwritten digits (28x28 grayscale images, 60K training, 10K testing)
# If use different dataset, then change class names and possibly normalization.
# mnist = keras.datasets.mnist
mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# Normalize inputs
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display first 25 training images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Reshape images for compatibility with convolutional layer
train_images = np.reshape(train_images, (60000, 28, 28, 1))
test_images = np.reshape(test_images, (10000, 28, 28, 1))

# Build, train, and evaluate model
def eval_model(model, epochs=10):
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    print("num", len(train_labels))
    hist = model.fit(train_images, train_labels, epochs=epochs)
    # train_acc = hist.history['acc'][-1]
    print("Evaluate/Test Model")
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    model.summary()
    print("Epochs: " + str(epochs))
    print(hist.history)
    # print('Training accuracy: ' + str(train_acc))
    # print('Testing accuracy: ' + str(test_acc))


# Model 1: flattened image input, 128-node dense ReLU hidden, 10-node dense softmax output


def run_model0(epochs=10):
    print(">>>>> Model 0:")  # Dense, Dense
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(units=32, activation=tf.nn.relu),
            keras.layers.Dense(units=32, activation=tf.nn.relu),
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    eval_model(model, epochs)
    print(">>>>> Model 0:")  # Dense, Dense


def run_model1(epochs=10):
    print(">>>>> Model 1:")  # Conv2D(64), MaxPool2D
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=(28, 28, 1),
                activation=tf.nn.relu,
                padding="same",
            ),  # 192, 21 filters; Added padding for fashion
            keras.layers.MaxPooling2D(pool_size=(2, 2)),  # pool_size = (3,3), (4,4)
            keras.layers.Flatten(),
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    eval_model(model, epochs)
    print(">>>>> Model 1:")  # Conv2D(64), MaxPool2D


def run_model2(epochs=10):
    print(">>>>> Model 2:")  # Conv2D(32), MaxPool2D, Conv2D(32), MaxPool2D
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(28, 28, 1),
                activation=tf.nn.relu,
                padding="same",
            ),  # 96, 10 filters; Added padding for fashion
            keras.layers.MaxPooling2D(pool_size=(2, 2)),  # pool_size = (3,3), (4,4)
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(28, 28, 1),
                activation=tf.nn.relu,
                padding="same",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    eval_model(model, epochs)
    print(">>>>> Model 2:")  # Conv2D(32), MaxPool2D, Conv2D(32), MaxPool2D


def run_model3(epochs=10):
    print(">>>>> Model 3:")  # Conv2D(64), MaxPool2D, Dense(64)
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=(28, 28, 1),
                activation=tf.nn.relu,
                padding="same",
            ),  # 192, 21 filters; Added padding for fashion
            keras.layers.MaxPooling2D(pool_size=(2, 2)),  # pool_size = (3,3), (4,4)
            keras.layers.Flatten(),
            keras.layers.Dense(units=21, activation=tf.nn.relu),  # 192, 21
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    eval_model(model, epochs)
    print(">>>>> Model 3:")  # Conv2D(64), MaxPool2D, Dense(64)


def run_model4(epochs=10):
    print(
        ">>>>> Model 4:"
    )  # Conv2D(32), MaxPool2D, Conv2D(32), MaxPool2D, Dense(32), Dense(32)
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(28, 28, 1),
                activation=tf.nn.relu,
                padding="same",
            ),  # 96, 10 filters; Added padding for fashion
            keras.layers.MaxPooling2D(pool_size=(2, 2)),  # pool_size = (3,3), (4,4)
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(28, 28, 1),
                activation=tf.nn.relu,
                padding="same",
            ),  # 96, 10 filters;
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(units=10, activation=tf.nn.relu),  # units = 96, 10;
            keras.layers.Dense(units=10, activation=tf.nn.relu),  # units = 96, 10;
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    eval_model(model, epochs)
    print(
        ">>>>> Model 4:"
    )  # Conv2D(32), MaxPool2D, Conv2D(32), MaxPool2D, Dense(32), Dense(32)


run_model0(epochs=10)
run_model1(epochs=10)
run_model2(epochs=10)
run_model3(epochs=10)
run_model4(epochs=10)
