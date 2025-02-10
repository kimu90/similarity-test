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
        """
        return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)))

    def calculate_jaccard_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Optimized Jaccard similarity calculation
        """
        # Pre-compute binary vectors using vectorized operations
        vec1_binary = vec1 > 0
        vec2_binary = vec2 > 0
        
        # Fast intersection and union using numpy logical operations
        intersection = np.logical_and(vec1_binary, vec2_binary).sum()
        union = np.logical_or(vec1_binary, vec2_binary).sum()
        
        return float(intersection / union if union > 0 else 0.0)

    def calculate_lcs_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate LCS similarity between vectors
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

    def batch_similarities(self, new_vecs: np.ndarray, true_vecs: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        """
        Calculate similarities for batches with optimized implementations
        """
        if metric == 'jaccard':
            # Pre-compute binary matrices once
            new_binary = new_vecs > 0
            true_binary = true_vecs > 0
            
            # Use matrix operations for all pairs at once
            intersection = new_binary @ true_binary.T
            new_sums = new_binary.sum(axis=1, keepdims=True)
            true_sums = true_binary.sum(axis=1)
            union = new_sums + true_sums[np.newaxis, :] - intersection
            
            # Avoid division by zero and return
            return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        
        elif metric == 'cosine':
            return cosine_similarity(new_vecs, true_vecs)
        
        elif metric == 'lcs':
            similarities = np.array([[
                self.calculate_lcs_similarity(v1, v2)
                for v2 in true_vecs
            ] for v1 in new_vecs])
            return similarities
        
        else:
            self.logger.error(f"Unknown similarity metric: {metric}")
            return np.array([])

    def get_true_embeddings_with_ids(self) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve TRUE set embeddings and their corresponding text IDs
        """
        try:
            query = """
                SELECT embedding, text_id 
                FROM embeddings 
                WHERE is_true = True
            """
            import pandas as pd
            df = pd.read_sql(query, self.db_manager.engine)
            
            if df.empty:
                self.logger.warning("No true embeddings found in database")
                return np.array([]), []
            
            embeddings = np.stack(df['embedding'].apply(np.array).values)
            text_ids = df['text_id'].tolist()
            
            return embeddings, text_ids
        except Exception as e:
            self.logger.error(f"Error retrieving TRUE embeddings: {str(e)}")
            return np.array([]), []

    def get_new_embeddings_with_ids(self) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve new embeddings and their corresponding text IDs
        """
        try:
            query = """
                SELECT embedding, text_id 
                FROM embeddings 
                WHERE is_true = False
            """
            import pandas as pd
            df = pd.read_sql(query, self.db_manager.engine)
            
            if df.empty:
                self.logger.warning("No new embeddings found in database")
                return np.array([]), []
            
            embeddings = np.stack(df['embedding'].apply(np.array).values)
            text_ids = df['text_id'].tolist()
            
            return embeddings, text_ids
        except Exception as e:
            self.logger.error(f"Error retrieving new embeddings: {str(e)}")
            return np.array([]), []

    def get_most_similar(self, 
                        similarities: np.ndarray, 
                        true_text_ids: List[str], 
                        new_text_ids: List[str],
                        k: int = 5) -> List[SimilarityResult]:
        """
        Get top k similar indices with scores
        """
        flat_similarities = similarities.flatten()
        top_k_indices = np.argsort(flat_similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
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
        """
        return min(1.0, similarity_score / self.similarity_threshold)

    def store_results(self, new_text_ids: List[str], similarities: np.ndarray, metric: str = 'cosine'):
        """
        Store similarity results
        """
        try:
            max_similarities = similarities.max(axis=1)
            
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
