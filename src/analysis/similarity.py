# src/analysis/similarity.py

import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass

@dataclass
class SimilarityResult:
   score: float
   index: int
   confidence: float

class SimilarityAnalyzer:
   def __init__(self, config: Dict):
       self.similarity_threshold = config.get('similarity_threshold', 0.8)
       self.logger = logging.getLogger(__name__)

   def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
       """Calculate cosine similarity between vectors"""
       return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)))

   def batch_similarities(self, new_vecs: np.ndarray, 
                        true_vecs: np.ndarray) -> np.ndarray:
       """Calculate similarities for batches"""
       return cosine_similarity(new_vecs, true_vecs)

   def get_most_similar(self, similarities: np.ndarray, 
                     k: int = 5) -> List[SimilarityResult]:
       """Get top k similar indices with scores"""
       top_k_idx = np.argsort(similarities)[-k:][::-1]
       results = []
       for idx in top_k_idx:
           score = similarities[idx]
           confidence = self._calculate_confidence(score)
           results.append(SimilarityResult(score, idx, confidence))
       return results

   def _calculate_confidence(self, similarity_score: float) -> float:
       """Calculate confidence score"""
       return min(1.0, similarity_score / self.similarity_threshold)