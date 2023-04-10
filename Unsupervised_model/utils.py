import numpy as np

def pairwise_distances(self, X):
    """
    Computes pairwise distances between samples in X.
    """
    distance = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            d = np.linalg.norm(X[i]-X[j])
            distance[i,j] = d
            distance[j,i] = d
    return distance

def normalize(self, X):
    """
    Normalizes the rows of X
    """
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X 

def probability(self, distance, perplexity):
    """
    Computes the P-values for the t-SNE algorithm.
    """
    n_samples = distance.shape[0]
    probability = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        beta = 1.0
        done = False
        while not done:
            p = np.exp(-distance[i] * beta)
            p[i] = 0.0
            sum_p = np.sum(p)
            if sum_p == 0.0:
                beta *= 2.0
            else:
                probability[i] = p / sum_p
                done = True
    probability = 0.5 * (probability + probability.T)
    probability = np.maximum(probability, np.finfo(float).eps)
    probability /= np.sum(probability)
    probability = np.maximum(probability, np.finfo(float).eps)
    return probability

def compute_q(self, Y):
    """
    Computes the Q-values for the tSNE algorithm.
    """
    n_samples = Y.shape[0]
    Q = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            q = 1.0 / (1.0 + np.linalg.norm(Y[i] - Y[j])**2)
            Q[i,j] = q
            Q[j,i] = q
    Q /= np.sum(Q)
    return Q

def compute_gradient(self, probability , gradient , Y):
    """
    Computes the gradients for the t-SNE algorithm.
    """
    pq_diff = probability - gradient

    # Compute pairwise distances in the embedded space

    Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distances = np.linalg.norm(Y_diff, axis=-1)

    # Compute the t-SNE gradient
    grad = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        grad[i] = 4 * np.sum(pq_diff[:, i, np.newaxis] * Y_diff[:, i] * (1 / (1 + distances[:, i]**2))[:, np.newaxis], axis=0)
    return grad

def cost(self, Y, probability):
    """
    Compute the t-SNE cost function.
    """
    # Compute pairwise distances in the embedded space
    distance = pairwise_distances(self,Y)
    
    # Compute the t-SNE cost function
    cost = np.sum(probability * np.log(probability / compute_q(self,Y)))  
    
    return cost