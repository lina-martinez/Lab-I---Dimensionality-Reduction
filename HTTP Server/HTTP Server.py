# Import necessary libraries

import numpy as np
import sys
sys.path.append(r'C:\Users\Lina\Documents\Lab-I---Dimensionality-Reduction\Unsupervised_model')

from SVD import SVD as MySVD
from PCA import PCA as MyPCA
from TSNE import TSNE as MyTSNE
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json

# Load the MNIST dataset
def load_mnist():
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist()

# Define a function to perform PCA, SVD, and t-SNE
def reduce_dimensionality(X, method):
    if method == "pca":
        print(X)
        pca = MyPCA(n_components=20)
        pca_fit = pca.fit(X)
        X_reduced = pca.fit_transform(X)
    elif method == "svd":
        svd = MySVD(n_components=20)
        svd_fit = svd.fit(X)
        X_reduced = svd.fit_transform(X)
    elif method == "tsne":
        tsne = MyTSNE(n_components=20,)
        tsne_fit = tsne.fit(X)
        X_reduced = tsne.transform(X)
    else:
        raise ValueError("Invalid dimensionality reduction method")
    return X_reduced

# Define a function to classify an input record using k-NN
def classify_record(record, k=5, method="pca"):
    X_reduced_train = reduce_dimensionality(X_train, method)
    X_reduced_record = reduce_dimensionality(np.expand_dims(record, axis=0), method)
    dists = np.linalg.norm(X_reduced_train - X_reduced_record, axis=1)
    closest_indices = np.argsort(dists)[:k]
    closest_labels = y_train[closest_indices]
    counts = np.bincount(closest_labels)
    label = np.argmax(counts)
    return int(label)

classify_record(X_train[0])

# Define a class that handles HTTP requests and responses
class MNISTDigitClassifierHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        record = np.fromstring(body, dtype=np.uint8).reshape((28, 224))
        label = classify_record(record)
        response = {"label": label}
        self._set_headers()
        self.wfile.write(json.dumps(response).encode())

# Start the HTTP server
def run_server(server_class=HTTPServer, handler_class=MNISTDigitClassifierHTTPRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}")
    httpd.serve_forever()

# run_server()






