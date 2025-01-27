# src/embeddings/centroid.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging
from config import config
from src.storage.db_manager import DatabaseManager

class CentroidCalculator:
    def __init__(self, config: Dict):
        """
        Initialize centroid calculator
        Args:
            config: Configuration with min vectors and other settings
        """
        self.min_vectors = config.get('min_vectors', 3)
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config)

    def load_true_embeddings(self) -> np.ndarray:
        """
        Load TRUE set embeddings from database
        Returns:
            Numpy array of TRUE set embeddings
        """
        try:
            embeddings = self.db_manager.get_true_embeddings()
            
            if len(embeddings) < self.min_vectors:
                raise ValueError(f"Need at least {self.min_vectors} vectors")
            
            self.logger.info(f"Loaded {len(embeddings)} TRUE set embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error loading TRUE set embeddings: {str(e)}")
            raise

    def calculate_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate centroid of embeddings
        Args:
            embeddings: Matrix of embeddings
        Returns:
            Centroid vector
        """
        return np.mean(embeddings, axis=0)

    def update_centroid(self, current_centroid: np.ndarray,
                        new_embedding: np.ndarray,
                        weight: float = 0.1) -> np.ndarray:
        """
        Update centroid with new embedding
        Args:
            current_centroid: Current centroid vector
            new_embedding: New embedding to incorporate
            weight: Weight for new embedding
        Returns:
            Updated centroid
        """
        return (1 - weight) * current_centroid + weight * new_embedding

    def get_distance_to_centroid(self, embedding: np.ndarray,
                                  centroid: np.ndarray) -> float:
        """
        Calculate cosine similarity to centroid
        Args:
            embedding: Input embedding
            centroid: Centroid vector
        Returns:
            Similarity score
        """
        return float(cosine_similarity(
            embedding.reshape(1, -1),
            centroid.reshape(1, -1)
        ))

    def get_closest_vectors(self, embeddings: np.ndarray,
                             centroid: np.ndarray,
                             k: int = 5) -> List[int]:
        """
        Get indices of k closest vectors to centroid
        Args:
            embeddings: Matrix of embeddings
            centroid: Centroid vector
            k: Number of closest vectors to return
        Returns:
            Indices of closest vectors
        """
        distances = np.array([
            self.get_distance_to_centroid(e, centroid)
            for e in embeddings
        ])
        return distances.argsort()[-k:][::-1]

if __name__ == "__main__":
    # Test the centroid calculator
    from src.preprocessing.loader import DataLoader

    # Initialize components
    loader = DataLoader()
    centroid_calc = CentroidCalculator(config)

    # Load TRUE set embeddings from database
    true_embeddings = centroid_calc.load_true_embeddings()

    # Calculate centroid
    centroid = centroid_calc.calculate_centroid(true_embeddings)

    # Get closest vectors
    closest_idx = centroid_calc.get_closest_vectors(true_embeddings, centroid, k=5)

    # Load original texts to print the most similar ones
    true_df = loader.load_true_set()

    print(f"Centroid shape: {centroid.shape}")
    print("\nMost similar texts to centroid:")
    for idx in closest_idx:
        similarity = centroid_calc.get_distance_to_centroid(true_embeddings[idx], centroid)
        print(f"Similarity: {similarity:.3f} - Text: {true_df['texts'].iloc[idx][:100]}...")