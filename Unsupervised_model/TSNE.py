import numpy as np

class TSNE:
    def __init__(self, n_components,perplexity=30, learning_rate=200, n_iters=1000, verbose=False):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)
        
    def fit_transform(self, X):
        # calcular las probabilidades condicionales y las entropías
        P = self._get_conditional_probabilities(X)
        H = self._get_entropy(P)
        
        # inicializar la solución aleatoriamente
        Y = np.random.normal(size=(X.shape[0], 2))
        
        # calcular la gradiente descendiente estocástica
        for i in range(self.n_iters):
            dY = self._get_gradient(Y, P, H)
            Y -= self.learning_rate * dY
            
            if self.verbose and i % 100 == 0:
                C = self._get_cost(P, Y)
                print(f"Iteration {i}, Cost: {C:.3f}")
                
        return Y
    
    def _get_conditional_probabilities(self, X):
        n_samples = X.shape[0]
        P = np.zeros((n_samples, n_samples))
        
        # calcular las distancias euclidianas entre las muestras
        distances = self._get_distances(X)
        
        for i in range(n_samples):
            # calcular las probabilidades de las muestras cercanas
            p = self._get_probabilities(distances[i], self.perplexity)
            
            # ajustar las probabilidades para que sean simétricas
            p[i] = 0
            P[i] = (p + p.T) / (2 * n_samples)
            
        # normalizar las probabilidades
        P = np.maximum(P, 1e-12)
        P /= np.sum(P)
        
        return P
    
    def _get_probabilities(self, distances, perplexity):
        # utilizar el método de bisección para encontrar la varianza
        tol = 1e-5
        var_min = 0
        var_max = np.inf
        var = 1
        while True:
            p = np.exp(-distances / (2 * var))
            p_sum = np.sum(p)
            h = np.log(p_sum) + var * np.sum(distances * p) / p_sum
            h_diff = h - np.log(perplexity)
            
            if np.abs(h_diff) < tol:
                break
            
            if h_diff > 0:
                var_max = var
                var = (var + var_min) / 2
            else:
                var_min = var
                if var_max == np.inf:
                    var = var * 2
                else:
                    var = (var + var_max) / 2
                    
        return p / p_sum
    
    def _get_entropy(self, P):
        H = -np.sum(P * np.log(P))
        return H
    
    def _get_distances(self, X):
        # calcular las distancias euclidianas entre las muestras
        XX = np.dot(X, X.T)
        X2 = np.sum(X ** 2, axis=1, keepdims=True)
        distances = X2 + X2.T - 2 * XX
        return distances
    
    def _get_gradient(self, Y, P, H):
        n_samples = Y.shape[0]
        Q = self._



