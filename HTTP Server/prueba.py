import requests
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the images to a 1D array and normalize the pixel values
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Send a POST request to the server with the first image in the training set
url = "http://localhost:8080/classify"
response = requests.post(url, data=X_train[0].tobytes())

# Print the predicted label
print(response.json()["label"])
