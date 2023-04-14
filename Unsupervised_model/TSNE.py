import numpy as np
import utils
from scipy.optimize import minimize


class TSNE:
    def __init__(self, n_components, perplexity=30, learning_rate=150.0, n_iter=100, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.mean = None
        self.random_state = random_state 
        self.embedding = None  

    def fit(self, X):
        X = np.asarray(X)
        samples = X.shape[0]
        features = X.shape[1]

        # Compute pairwise distances
        distance = utils.pairwise_distances(self,X)

        # Compute conditional probabilities
        probability = utils.probability(self,distance, self.perplexity)

        # Initialize Y randomly
        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(size=(X.shape[0], self.n_components))

        # Optimization using gradient descent
        for i in range(self.n_iter):
            # Calculate dY

            # Compute Q-values
            Q = utils.compute_q(self,Y)
            
            # Compute gradients
            grad = utils.compute_gradient(self,probability, Q, Y, distance)

            # Update Y
            Y -= self.learning_rate * grad

        self.embedding = Y 
    
    def transform(self, X):
        X = np.asarray(X)
        distance = utils.pairwise_distances(self,X)
        probability = utils.probability(self,distance, self.perplexity)
        Q = utils.compute_q(self,self.embedding)
        return utils.compute_gradient(self,probability, Q,self.embedding,distance)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.embedding

# import matplotlib.pyplot as plt
# from sklearn.datasets import load_wine
# from sklearn.preprocessing import StandardScaler

# # Load DataSet
# wine_data = load_wine()
# X, y = wine_data['data'], wine_data['target']

# X_scaled = StandardScaler().fit_transform(X)

# # Create a PCA robust object with 2 components
# PCArobust = TSNE(n_components=2)
# # Fit the data and transform
# PCArobust_fit = PCArobust.fit(X_scaled)
# PCArobust_transform = PCArobust.transform(X_scaled)

# print(PCArobust_transform.shape)

# plot = plt.scatter(PCArobust_transform[:,0], PCArobust_transform[:,1], c=y)
# plt.legend(handles=plot.legend_elements()[0], labels=list(wine_data['target_names']))
# plt.show()

