import numpy as np
import utils

class TSNE:
    def __init__(self, n_components, perplexity=30, learning_rate=200.0, n_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state 
        self.embedding = None  

    def fit(self, X):
        X = np.asarray(X)
        samples = X.shape
        features = X.shape

        # Compute pairwise distances
        distance = self.utils.pairwise_distances(X)
        
        # Compute conditional probabilities
        probability = self.utils.probability(distance, self.perplexity)
        
        # Initialize Y randomly
        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(size=(X.shape[0], self.n_components))
        Y = self.utils.normalize(Y)
        
        # Optimization using gradient descent
        for i in range(self.n_iter):
            # Calculate dY

            # Compute Q-values
            Q = self.utils.compute_q(Y)
            
            # Compute gradients
            grad = self.utils.compute_gradient(probability, Q, Y)

            # Update Y
            Y -= self.learning_rate * grad
            Y = self.utils.normalize(Y)
    
        self.embedding = Y 
    
    def transform(self, X):
        X = np.asarray(X)
        distance = self.utils.pairwise_distances(X)
        probability = self.utils.probability(distance, self.perplexity)
        Q = self.utils.compute_q(self.embedding)
        return self.utils.compute_gradient(probability, Q,self.embedding)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.embedding

    def inverse_transform(self, Y):
        Y = np.asarray(Y)
        X = self.embedding # Use the original embedding as a starting point
        n_iter = 50 # Use a smaller number of iterations for the inverse transform
        for i in range(n_iter):
            Q = self.utils.compute_q(Y)
            grad = self.utils.compute_gradient(Q, self.utils.probability(self.utils.pairwise_distances(X), self.perplexity), X)
            X -= self.learning_rate * grad
            X = self.utils.normalize(X)
        return X


    

