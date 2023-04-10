import numpy as np
import utils
from scipy.optimize import minimize


class TSNE:
    def __init__(self, n_components, perplexity=30, learning_rate=150.0, n_iter=10, random_state=None):
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
            print(self.n_iter)
            # Compute Q-values
            Q = utils.compute_q(self,Y)
            
            # Compute gradients
            grad = utils.compute_gradient(self,probability, Q, Y)

            # Update Y
            Y -= self.learning_rate * grad

        self.embedding = Y 
    
    def transform(self, X):
        X = np.asarray(X)
        distance = utils.pairwise_distances(self,X)
        probability = utils.probability(self,distance, self.perplexity)
        Q = utils.compute_q(self,self.embedding)
        return utils.compute_gradient(self,probability, Q,self.embedding)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.embedding

    def inverse_transform(self, Y):
        Y = np.asarray(Y)
        X = self.embedding # Use the original embedding as a starting point
        n_iter = 50 # Use a smaller number of iterations for the inverse transform
        for i in range(n_iter):
            Q = utils.compute_q(self,Y)
            grad = utils.compute_gradient(self,Q, self.utils.probability(self.utils.pairwise_distances(X), self.perplexity), X)
            X -= self.learning_rate * grad
            X = utils.normalize(self,X)
        return X


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

