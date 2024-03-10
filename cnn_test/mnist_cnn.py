import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import os

# Dataset
data_dir = './MNIST_ORG'

# Load training data
with open(os.path.join(data_dir, 'train-images.idx3-ubyte'), 'rb') as f:
    images_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

with open(os.path.join(data_dir, 'train-labels.idx1-ubyte'), 'rb') as f:
    labels_train = np.frombuffer(f.read(), np.uint8, offset=8)

# Load test data
with open(os.path.join(data_dir, 't10k-images.idx3-ubyte'), 'rb') as f:
    images_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

with open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'), 'rb') as f:
    labels_test = np.frombuffer(f.read(), np.uint8, offset=8)


# Reshape and normalize input data
x_train = images_train.reshape(-1, 28, 28, 1) / 255.0
x_test = images_test.reshape(-1, 28, 28, 1) / 255.0
y_train = labels_train
y_test = labels_test

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))


# Evaluate model on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

