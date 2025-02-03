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

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
        
        Returns:
            Similarity score
        """
        return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)))

    def compute_correlation_matrix(self):
        """
        Compute a correlation matrix between true set and new texts
        
        Returns:
            np.ndarray: Correlation matrix of similarities
        """
        try:
            # Combine texts from both datasets
            all_texts = pd.concat([
                self.true_set_df['text'], 
                self.new_texts_df['text']
            ])

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            
            # Compute TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Compute cosine similarity matrix
            correlation_matrix = cosine_similarity(tfidf_matrix)
            
            # Store the matrix as an instance attribute
            self.correlation_matrix = correlation_matrix
            
            return correlation_matrix
        
        except Exception as e:
            print(f"Error computing correlation matrix: {e}")
            return None

    def calculate_jaccard_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Jaccard similarity between vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
        
        Returns:
            Similarity score
        """
        # Convert vectors to binary (presence/absence)
        vec1_binary = (vec1 > 0).astype(int)
        vec2_binary = (vec2 > 0).astype(int)
        
        intersection = np.sum(np.minimum(vec1_binary, vec2_binary))
        union = np.sum(np.maximum(vec1_binary, vec2_binary))
        
        return float(intersection / union if union > 0 else 0.0)

    def calculate_lcs_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate LCS similarity between vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
        
        Returns:
            Similarity score
        """
        # Convert vectors to binary for sequence comparison
        vec1_binary = (vec1 > 0).astype(int)
        vec2_binary = (vec2 > 0).astype(int)
        
        m, n = len(vec1_binary), len(vec2_binary)
        matrix = np.zeros((m + 1, n + 1))
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if vec1_binary[i-1] == vec2_binary[j-1]:
                    matrix[i,j] = matrix[i-1,j-1] + 1
                else:
                    matrix[i,j] = max(matrix[i-1,j], matrix[i,j-1])
        
        lcs_length = matrix[m,n]
        return float((2.0 * lcs_length) / (m + n) if (m + n) > 0 else 0.0)

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
            
            # Handle case where no true embeddings exist
            if df.empty:
                self.logger.warning("No true embeddings found in database")
                return np.array([]), []
            
            # Convert embeddings to numpy array
            embeddings = np.stack(df['embedding'].apply(np.array).values)
            text_ids = df['text_id'].tolist()
            
            return embeddings, text_ids
        except Exception as e:
            self.logger.error(f"Error retrieving TRUE embeddings: {str(e)}")
            return np.array([]), []

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
            
            # Handle case where no new embeddings exist
            if df.empty:
                self.logger.warning("No new embeddings found in database")
                return np.array([]), []
            
            # Convert embeddings to numpy array
            embeddings = np.stack(df['embedding'].apply(np.array).values)
            text_ids = df['text_id'].tolist()
            
            return embeddings, text_ids
        except Exception as e:
            self.logger.error(f"Error retrieving new embeddings: {str(e)}")
            return np.array([]), []

    def batch_similarities(self, new_vecs: np.ndarray, true_vecs: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        """
        Calculate similarities for batches
        
        Args:
            new_vecs: Matrix of new embeddings
            true_vecs: Matrix of TRUE set embeddings
            metric: Which similarity metric to use ('cosine', 'jaccard', or 'lcs')
        
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
            # Fast path for cosine similarity
            if metric == 'cosine':
                similarities = cosine_similarity(new_vecs, true_vecs)
                
            # Calculate Jaccard similarity if requested
            elif metric == 'jaccard':
                similarities = np.array([[
                    self.calculate_jaccard_similarity(v1, v2)
                    for v2 in true_vecs
                ] for v1 in new_vecs])
                
            # Calculate LCS similarity if requested
            elif metric == 'lcs':
                similarities = np.array([[
                    self.calculate_lcs_similarity(v1, v2)
                    for v2 in true_vecs
                ] for v1 in new_vecs])
                
            else:
                self.logger.error(f"Unknown similarity metric: {metric}")
                return np.array([])
            
            self.logger.info(f"Calculated {metric} similarities: shape {similarities.shape}")
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
            similarities: Matrix of similarity scores
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

    def store_results(self, new_text_ids: List[str], similarities: np.ndarray, metric: str = 'cosine'):
        """
        Store similarity results
        
        Args:
            new_text_ids: List of text IDs
            similarities: Matrix of similarity scores
            metric: Similarity metric used
        """
        try:
            max_similarities = similarities.max(axis=1)
            
            # Prepare results for database storage
            results = [
                {
                    'text_id': text_id,
                    'similarity_score': float(score),
                    'metric': metric,
                    'confidence': float(self._calculate_confidence(score)),
                    'label': bool(score >= self.similarity_threshold)
                }
                for text_id, score in zip(new_text_ids, max_similarities)
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

    # Test each similarity metric
    for metric in ['cosine', 'jaccard', 'lcs']:
        print(f"\nTesting {metric} similarity:")
        
        # Calculate similarities
        similarities = similarity.batch_similarities(
            new_embeddings, 
            true_embeddings,
            metric=metric
        )

        # Get most similar texts
        results = similarity.get_most_similar(
            similarities, 
            true_text_ids, 
            new_text_ids, 
            k=3
        )

        # Store results
        similarity.store_results(new_text_ids, similarities, metric)

        # Print results
        for result in results:
            print(f"\nSimilarity: {result.score:.3f}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Text ID: {result.text_id}")