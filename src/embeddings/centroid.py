# src/embeddings/centroid.py

import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

class CentroidCalculator:
   def __init__(self, config: Dict):
       self.min_vectors = config.get('min_vectors', 3)
       self.logger = logging.getLogger(__name__)
       
   def calculate_centroid(self, embeddings: np.ndarray) -> np.ndarray:
       """Calculate centroid of TRUE set embeddings"""
       if len(embeddings) < self.min_vectors:
           raise ValueError(f"Need at least {self.min_vectors} vectors")
       return np.mean(embeddings, axis=0)

   def update_centroid(self, current_centroid: np.ndarray, 
                      new_embedding: np.ndarray,
                      weight: float = 0.1) -> np.ndarray:
       """Update centroid with new embedding"""
       return (1 - weight) * current_centroid + weight * new_embedding

   def get_distance_to_centroid(self, embedding: np.ndarray, 
                              centroid: np.ndarray) -> float:
       """Calculate cosine similarity to centroid"""
       return float(cosine_similarity(
           embedding.reshape(1, -1), 
           centroid.reshape(1, -1)
       ))

   def get_closest_vectors(self, embeddings: np.ndarray,
                         centroid: np.ndarray,
                         k: int = 5) -> List[int]:
       """Get indices of k closest vectors to centroid"""
       distances = np.array([
           self.get_distance_to_centroid(e, centroid) 
           for e in embeddings
       ])
       return distances.argsort()[-k:][::-1]