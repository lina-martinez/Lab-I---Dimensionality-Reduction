import numpy as np
import sys
sys.path.append(r'C:\Users\Lina\Documents\Lab-I---Dimensionality-Reduction\Unsupervised_model')

from SVD import SVD as MySVD
from PCA import PCA as MyPCA
from TSNE import TSNE as MyTSNE
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the MNIST dataset
# mnist = fetch_openml('mnist_784', version=1)
# mnist.target = mnist.target.astype(int)
iris = load_iris()
# Dividir datos en X e Y
X = iris.data
y = iris.target

# Select only the images corresponding to 0s and 8s
# X = mnist.data[(mnist.target == 0) | (mnist.target == 8)]
# y = mnist.target[(mnist.target == 0) | (mnist.target == 8)]

pca = MyTSNE(n_components=2)
X_svd_fit = pca.fit(X)
X_transformed = pca.fit_transform(X)

# Convert to real values
X_transformed = np.real(X_transformed)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Entrenar un clasificador logístico en los datos de PCA
clf = LogisticRegression(penalty = 'none', random_state=0, max_iter=1000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

# Print the predicted cluster and actual label of the new record
print("Predicted cluster:", pred[0])
print("Actual label:", y_test[0])

accuracy = accuracy_score(y_test, pred)
print(accuracy)