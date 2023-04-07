import numpy as np

class PCA:

    def __init__(self, n_components):
        #inicializa la clase con el número de componentes principales que se desean obtener.
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)

    def transform(self, X):
        X = X - self.mean

        # calcular la matriz de covarianza
        cov_matrix = np.cov(X, rowvar=False)
        
        # calcular los autovalores y autovectores de la matriz de covarianza
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # ordenar los autovalores y autovectores de mayor a menor
        sorted_indexes = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indexes]
        eigenvectors = eigenvectors[:, sorted_indexes]
        
        # seleccionar los primeros n_componentes autovectores
        self.components = eigenvectors[:, :self.n_components]
    
    def fit_transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed
    
    def inverse_transform(self, X): 
        X_reconstructed = (X @ self.components).dot(self.components.T) + np.mean(X, axis=0)      
        return X_reconstructed