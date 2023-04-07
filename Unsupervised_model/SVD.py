import numpy as np

class SVD:

    def __init__(self, n_components):
        #inicializa la clase con el número de componentes principales que se desean obtener.
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)
        
    def transform(self, X):
        # proyectar los datos en los nuevos componentes
        X = X - self.mean

        #compute the vectors
        self.U, self.S, self.VT = np.linalg.svd(X) 

        #compute mean and std to standarization    
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)

        if self.n_components is not None:
                self.U = self.U[:, :self.n_components]
                self.S = self.S[:self.n_components]
                self.VT = self.VT[:self.n_components, :]
        self.components = self.VT[:self.n_components].T
    
    def fit_transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed
    
    def inverse_transform(self, X): 
        X_reconstructed = (X @ self.components).dot(self.components.T) + np.mean(X, axis=0)      
        return X_reconstructed
