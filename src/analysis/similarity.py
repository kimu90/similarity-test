# src/analysis/similarity.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from config import config

@dataclass
class SimilarityResult:
    score: float
    index: int
    confidence: float
    text_id: str

class SimilarityAnalyzer:
    def __init__(self, config: Dict, db_manager=None):
        """
        Initialize similarity analyzer
        
        Args:
            config: Configuration with similarity threshold settings
            db_manager: Optional database manager
        """
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.logger = logging.getLogger(__name__)
        
        # Lazy import of DatabaseManager to avoid circular imports
        if db_manager is None:
            from src.storage.db_manager import DatabaseManager
            db_manager = DatabaseManager(config)
        
        self.db_manager = db_manager

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
        try:
            query = """
                SELECT embedding, text_id 
                FROM embeddings 
                WHERE is_true = True
            """
            import pandas as pd
            df = pd.read_sql(query, self.db_manager.engine)
            
            # Convert embeddings to numpy array
            embeddings = np.stack(df['embedding'].apply(np.array).values)
            text_ids = df['text_id'].tolist()
            
            return embeddings, text_ids
        except Exception as e:
            self.logger.error(f"Error retrieving TRUE embeddings: {str(e)}")
            raise

    def get_new_embeddings_with_ids(self) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve new embeddings and their corresponding text IDs
        
        Returns:
            Tuple of embeddings array and list of text IDs
        """
        try:
            query = """
                SELECT embedding, text_id 
                FROM embeddings 
                WHERE is_true = False
            """
            import pandas as pd
            df = pd.read_sql(query, self.db_manager.engine)
            
            # Convert embeddings to numpy array
            embeddings = np.stack(df['embedding'].apply(np.array).values)
            text_ids = df['text_id'].tolist()
            
            return embeddings, text_ids
        except Exception as e:
            self.logger.error(f"Error retrieving new embeddings: {str(e)}")
            raise

    def batch_similarities(self, new_vecs: np.ndarray, true_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate similarities for batches
        
        Args:
            new_vecs: Matrix of new embeddings
            true_vecs: Matrix of TRUE set embeddings
        
        Returns:
            Matrix of similarity scores
        """
        # Add logging and error checking
        if new_vecs.size == 0:
            self.logger.error("No new embeddings provided")
            return np.array([])
        
        if true_vecs.size == 0:
            self.logger.error("No TRUE set embeddings provided")
            return np.array([])
        
        # Ensure 2D arrays
        if new_vecs.ndim == 1:
            new_vecs = new_vecs.reshape(1, -1)
        
        if true_vecs.ndim == 1:
            true_vecs = true_vecs.reshape(1, -1)
        
        try:
            similarities = cosine_similarity(new_vecs, true_vecs)
            self.logger.info(f"Calculated similarities: shape {similarities.shape}")
            return similarities
        except Exception as e:
            self.logger.error(f"Error calculating similarities: {str(e)}")
            return np.array([])

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
        """
        Store similarity results in database
        
        Args:
            new_text_ids: List of new text identifiers
            similarities: Similarity matrix
        """
        try:
            # Lazy import to avoid circular dependency
            from src.analysis.classifier import TextClassifier
            
            classifier = TextClassifier(config, self.db_manager)
            max_similarities = similarities.max(axis=1)
            classifications = classifier.batch_classify(max_similarities, new_text_ids)
            
            # Convert ClassificationResult to dictionary for storage
            results = [
                {
                    'text_id': result.text_id,
                    'similarity': result.similarity_score,
                    'confidence': result.confidence,
                    'label': result.label
                }
                for result in classifications
            ]
            
            self.db_manager.store_results(results)
            self.logger.info(f"Stored {len(results)} similarity results")
        
        except Exception as e:
            self.logger.error(f"Error storing similarity results: {str(e)}")
            raise

if __name__ == "__main__":
    # Test similarity analyzer
    from src.storage.db_manager import DatabaseManager

    # Initialize components
    db_manager = DatabaseManager(config)
    similarity = SimilarityAnalyzer(config, db_manager)

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
    similarity.store_results(new_text_ids, similarities)

    # Print results
    for result in results:
        print(f"\nSimilarity: {result.score:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Text ID: {result.text_id}")