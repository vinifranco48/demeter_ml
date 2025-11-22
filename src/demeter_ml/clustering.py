from sklearn.cluster import KMeans
import numpy as np
import pickle
import os

class GrainClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_fitted = False

    def train(self, features: np.ndarray):
        """
        Trains the K-Means model on the provided features.
        """
        self.model.fit(features)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster for the provided features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict(features)

    def save(self, path: str):
        """
        Saves the trained model to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """
        Loads a trained model from a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
