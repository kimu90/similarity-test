# src/analysis/similarity.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from config import config

@dataclass
class SimilarityResult:
   score: float
   index: int
   confidence: float

class SimilarityAnalyzer:
   def __init__(self, config: Dict):
       """
       Initialize similarity analyzer
       Args:
           config: Configuration with similarity threshold settings
       """
       self.similarity_threshold = config.get('similarity_threshold', 0.8)
       self.logger = logging.getLogger(__name__)

   def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
       """
       Calculate cosine similarity between vectors
       Args:
           vec1: First embedding vector
           vec2: Second embedding vector
       Returns:
           Similarity score
       """
       return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)))

   def batch_similarities(self, new_vecs: np.ndarray, 
                        true_vecs: np.ndarray) -> np.ndarray:
       """
       Calculate similarities for batches
       Args:
           new_vecs: Matrix of new embeddings
           true_vecs: Matrix of TRUE set embeddings
       Returns:
           Matrix of similarity scores
       """
       return cosine_similarity(new_vecs, true_vecs)

   def get_most_similar(self, similarities: np.ndarray, 
                     k: int = 5) -> List[SimilarityResult]:
       """
       Get top k similar indices with scores
       Args:
           similarities: Array of similarity scores
           k: Number of top results to return
       Returns:
           List of similarity results
       """
       top_k_idx = np.argsort(similarities)[-k:][::-1]
       results = []
       for idx in top_k_idx:
           score = similarities[idx]
           confidence = self._calculate_confidence(score)
           results.append(SimilarityResult(score, idx, confidence))
       return results

   def _calculate_confidence(self, similarity_score: float) -> float:
       """
       Calculate confidence score based on similarity
       Args:
           similarity_score: Raw similarity score
       Returns:
           Confidence score
       """
       return min(1.0, similarity_score / self.similarity_threshold)

if __name__ == "__main__":
   # Test similarity analyzer
   from src.preprocessing.loader import DataLoader
   from src.preprocessing.cleaner import TextPreprocessor
   from src.embeddings.generator import EmbeddingGenerator
   from src.embeddings.centroid import CentroidCalculator

   # Initialize components
   loader = DataLoader()
   cleaner = TextPreprocessor(config)
   generator = EmbeddingGenerator(config)
   similarity = SimilarityAnalyzer(config)

   # Load and process data
   true_df = loader.load_true_set()
   new_df = loader.load_new_texts(batch_size=100)

   cleaned_true = cleaner.batch_process(true_df['texts'].tolist())
   cleaned_new = cleaner.batch_process(new_df['texts'].tolist())

   # Generate embeddings
   true_embeddings = generator.batch_generate(cleaned_true)
   new_embeddings = generator.batch_generate(cleaned_new)

   # Calculate similarities
   similarities = similarity.batch_similarities(new_embeddings, true_embeddings)
   
   # Get most similar texts
   print("\nMost similar texts:")
   for sim_batch in similarities:
       top_similar = similarity.get_most_similar(sim_batch, k=3)
       for result in top_similar:
           print(f"\nSimilarity: {result.score:.3f}")
           print(f"Confidence: {result.confidence:.3f}")
           print(f"TRUE text: {true_df['texts'].iloc[result.index][:100]}...")