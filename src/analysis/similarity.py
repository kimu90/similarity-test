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
from src.storage.db_manager import DatabaseManager

@dataclass
class SimilarityResult:
    score: float
    index: int
    confidence: float
    text_id: str

class SimilarityAnalyzer:
    def __init__(self, config: Dict):
        """
        Initialize similarity analyzer
        Args:
            config: Configuration with similarity threshold settings
        """
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config)

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

    def get_true_embeddings_with_ids(self) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve TRUE set embeddings and their corresponding text IDs
        Returns:
            Tuple of embeddings array and list of text IDs
        """
        # Custom query to get both embeddings and text_ids
        query = """
            SELECT embedding, text_id 
            FROM embeddings 
            WHERE is_true = True
        """
        import pandas as pd
        df = pd.read_sql(query, self.db_manager.conn)
        
        # Convert embeddings to numpy array
        embeddings = np.stack(df['embedding'].values)
        text_ids = df['text_id'].tolist()
        
        return embeddings, text_ids

    def get_new_embeddings_with_ids(self) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve new embeddings and their corresponding text IDs
        Returns:
            Tuple of embeddings array and list of text IDs
        """
        # Custom query to get both embeddings and text_ids
        query = """
            SELECT embedding, text_id 
            FROM embeddings 
            WHERE is_true = False
        """
        import pandas as pd
        df = pd.read_sql(query, self.db_manager.conn)
        
        # Convert embeddings to numpy array
        embeddings = np.stack(df['embedding'].values)
        text_ids = df['text_id'].tolist()
        
        return embeddings, text_ids

    def batch_similarities(self, new_vecs: np.ndarray, true_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate similarities for batches
        Args:
            new_vecs: Matrix of new embeddings
            true_vecs: Matrix of TRUE set embeddings
        Returns:
            Matrix of similarity scores
        """
        return cosine_similarity(new_vecs, true_vecs)

    def get_most_similar(self, 
                          similarities: np.ndarray, 
                          true_text_ids: List[str], 
                          new_text_ids: List[str],
                          k: int = 5) -> List[SimilarityResult]:
        """
        Get top k similar indices with scores
        Args:
            similarities: Array of similarity scores
            true_text_ids: List of TRUE set text IDs
            new_text_ids: List of new text IDs
            k: Number of top results to return
        Returns:
            List of similarity results
        """
        # Flatten the similarities matrix
        flat_similarities = similarities.flatten()
        
        # Get indices of top k similarities
        top_k_indices = np.argsort(flat_similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            # Convert flat index to 2D matrix index
            new_idx = idx // len(true_text_ids)
            true_idx = idx % len(true_text_ids)
            
            score = similarities[new_idx, true_idx]
            confidence = self._calculate_confidence(score)
            
            results.append(SimilarityResult(
                score=score, 
                index=true_idx, 
                confidence=confidence,
                text_id=true_text_ids[true_idx]
            ))
        
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

    def store_results(self, new_text_ids: List[str], similarities: np.ndarray):
        max_similarities = similarities.max(axis=1)
        classifications = self.batch_classify(max_similarities, new_text_ids)
        
        # Convert ClassificationResult to dictionary for storage
        results = [
            {
                'text_id': result.lens_id,
                'similarity': result.similarity_score,
                'confidence': result.confidence,
                'label': result.label
            }
            for result in classifications
        ]
        
        self.db_manager.store_results(results)
if __name__ == "__main__":
    # Test similarity analyzer
    similarity = SimilarityAnalyzer(config)

    # Retrieve embeddings from database
    true_embeddings, true_text_ids = similarity.get_true_embeddings_with_ids()
    new_embeddings, new_text_ids = similarity.get_new_embeddings_with_ids()

    # Calculate similarities
    similarities = similarity.batch_similarities(new_embeddings, true_embeddings)

    # Get most similar texts
    print("\nMost similar texts:")
    results = similarity.get_most_similar(
        similarities, 
        true_text_ids, 
        new_text_ids, 
        k=3
    )

    # Store results in database
    similarity.store_results(results, new_text_ids)

    # Print results
    for result in results:
        print(f"\nSimilarity: {result.score:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Text ID: {result.text_id}")