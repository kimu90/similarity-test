import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
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

    def _generate_character_ngrams(self, text: str, n: int) -> Set[str]:
        """
        Generate character-level n-grams from text
        
        Parameters:
        -----------
        text : str
            Input text
        n : int
            Size of n-grams
            
        Returns:
        --------
        Set[str]
            Set of character n-grams
        """
        # Pad the text for edge n-grams
        padded_text = ' ' * (n-1) + text + ' ' * (n-1)
        return set(padded_text[i:i+n] for i in range(len(padded_text)-n+1))

    def _generate_word_ngrams(self, text: str, n: int) -> Set[str]:
        """
        Generate word-level n-grams from text
        
        Parameters:
        -----------
        text : str
            Input text
        n : int
            Size of n-grams
            
        Returns:
        --------
        Set[str]
            Set of word n-grams
        """
        words = text.split()
        if len(words) < n:
            return set()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    def calculate_jaccard_similarity(self, text1, text2, n: int = 3, 
                                   use_char_ngrams: bool = True, 
                                   use_word_ngrams: bool = True,
                                   char_weight: float = 0.5) -> float:
        """
        Calculate weighted Jaccard similarity using both character and word n-grams
        
        Parameters:
        -----------
        text1 : str or np.ndarray
            First input text or vector
        text2 : str or np.ndarray
            Second input text or vector
        n : int, optional
            Size of n-grams to use. Default is 3 (trigrams)
        use_char_ngrams : bool, optional
            Whether to use character-level n-grams. Default is True
        use_word_ngrams : bool, optional
            Whether to use word-level n-grams. Default is True
        char_weight : float, optional
            Weight for character-level n-grams (1 - char_weight for word n-grams).
            Default is 0.5 (equal weighting)
            
        Returns:
        --------
        float
            Weighted Jaccard similarity score between 0 and 1
        """
        # Input validation
        if not use_char_ngrams and not use_word_ngrams:
            raise ValueError("At least one of use_char_ngrams or use_word_ngrams must be True")
        if not 0 <= char_weight <= 1:
            raise ValueError("char_weight must be between 0 and 1")
        
        # Convert numpy arrays to strings if needed
        if isinstance(text1, np.ndarray):
            text1 = ' '.join(map(str, text1))
        if isinstance(text2, np.ndarray):
            text2 = ' '.join(map(str, text2))
        
        # Normalize texts
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        # Special cases
        if text1 == text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        similarity = 0.0
        weights_sum = 0.0
        
        # Character n-grams similarity
        if use_char_ngrams:
            char_ngrams1 = self._generate_character_ngrams(text1, n)
            char_ngrams2 = self._generate_character_ngrams(text2, n)
            
            if char_ngrams1 and char_ngrams2:
                char_intersection = len(char_ngrams1.intersection(char_ngrams2))
                char_union = len(char_ngrams1.union(char_ngrams2))
                char_similarity = char_intersection / char_union if char_union > 0 else 0.0
                similarity += char_weight * char_similarity
                weights_sum += char_weight
        
        # Word n-grams similarity
        if use_word_ngrams:
            word_ngrams1 = self._generate_word_ngrams(text1, n)
            word_ngrams2 = self._generate_word_ngrams(text2, n)
            
            if word_ngrams1 and word_ngrams2:
                word_intersection = len(word_ngrams1.intersection(word_ngrams2))
                word_union = len(word_ngrams1.union(word_ngrams2))
                word_similarity = word_intersection / word_union if word_union > 0 else 0.0
                word_weight = 1.0 - char_weight
                similarity += word_weight * word_similarity
                weights_sum += word_weight
        
        # Normalize by actual weights used
        final_similarity = similarity / weights_sum if weights_sum > 0 else 0.0
        return float(max(0.0, min(1.0, final_similarity)))

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

    def calculate_levenshtein_similarity(self, text1, text2) -> float:
        """
        Calculate similarity based on Levenshtein distance
        """
        # Convert numpy arrays to strings if needed
        if isinstance(text1, np.ndarray):
            text1 = ' '.join(map(str, text1))
        if isinstance(text2, np.ndarray):
            text2 = ' '.join(map(str, text2))
        
        # Normalize texts
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        # Special cases
        if text1 == text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Levenshtein distance calculation
        def levenshtein_distance(s1, s2):
            # Ensure s1 is the longer string
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            # If one string is empty, distance is length of the other
            if len(s2) == 0:
                return len(s1)
            
            # Previous and current row of distances
            previous_row = range(len(s2) + 1)
            
            # Iterate through s1
            for i, c1 in enumerate(s1):
                # Create a new row of distances
                current_row = [i + 1]
                
                # Compute distances for this row
                for j, c2 in enumerate(s2):
                    # Insertions, deletions, substitutions
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    
                    current_row.append(min(insertions, deletions, substitutions))
                
                # Update previous row for next iteration
                previous_row = current_row
            
            # Last element is the Levenshtein distance
            return previous_row[-1]
        
        # Compute Levenshtein distance
        distance = levenshtein_distance(text1, text2)
        max_length = max(len(text1), len(text2))
        
        # Convert to similarity score
        similarity = 1 - (distance / max_length)
        
        return max(0.0, min(1.0, similarity))

    def calculate_euclidean_similarity(self, text1, text2) -> float:
        """
        Calculate Euclidean-inspired similarity for short text sequences
        """
        # Convert numpy arrays to strings if needed
        if isinstance(text1, np.ndarray):
            text1 = ' '.join(map(str, text1))
        if isinstance(text2, np.ndarray):
            text2 = ' '.join(map(str, text2))
        
        # Normalize texts
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        # Special cases
        if text1 == text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Calculate Levenshtein distance
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            # Length difference as a penalty
            if len(s1) > len(s2):
                s1 = s1[:len(s2)]
            
            distances = range(len(s1) + 1)
            for i2, c2 in enumerate(s2):
                distances_ = [i2+1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        distances_.append(distances[i1])
                    else:
                        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
                distances = distances_
            return distances[-1]
        
        # Calculate character-level overlap
        def char_overlap(s1, s2):
            set1 = set(s1)
            set2 = set(s2)
            return len(set1.intersection(set2)) / len(set1.union(set2))
        
        # Compute Levenshtein distance
        distance = levenshtein_distance(text1, text2)
        max_length = max(len(text1), len(text2))
        
        # Normalize distance
        normalized_distance = 1 - (distance / max_length)
        
        # Add character overlap as a bonus
        char_similarity = char_overlap(text1, text2)
        
        # Combine distance normalization and character overlap
        similarity = 0.7 * normalized_distance + 0.3 * char_similarity
        
        return max(0.0, min(1.0, similarity))

    def batch_similarities(self, new_texts, true_texts, metric: str = 'cosine', **kwargs) -> np.ndarray:
        """
        Calculate similarities for batches with optimized implementations
        
        Parameters:
        -----------
        new_texts : array-like
            Array of new texts or vectors to compare
        true_texts : array-like
            Array of true texts or vectors to compare against
        metric : str, optional
            Similarity metric to use. Default is 'cosine'
        **kwargs : dict
            Additional parameters for specific metrics:
            - n : int
                Size of n-grams for Jaccard similarity (default: 3)
            - use_char_ngrams : bool
                Whether to use character n-grams for Jaccard (default: True)
            - use_word_ngrams : bool
                Whether to use word n-grams for Jaccard (default: True)
            - char_weight : float
                Weight for character n-grams in Jaccard (default: 0.5)
        
        Returns:
        --------
        np.ndarray
            2D array of similarity scores
        """
        # Ensure inputs are numpy arrays
        new_texts = np.asarray(new_texts)
        true_texts = np.asarray(true_texts)
        
        # Check if inputs are numerical
        if new_texts.dtype.kind in 'biufc' and true_texts.dtype.kind in 'biufc':
            # Numerical vector similarity
            if metric == 'jaccard':
                # Pre-compute binary matrices once
                new_binary = new_texts > 0
                true_binary = true_texts > 0
                
                # Use matrix operations for all pairs at once
                intersection = new_binary @ true_binary.T
                new_sums = new_binary.sum(axis=1, keepdims=True)
                true_sums = true_binary.sum(axis=1)
                union = new_sums + true_sums[np.newaxis, :] - intersection
                
                # Avoid division by zero and return
                return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
            
            elif metric == 'cosine':
                return cosine_similarity(new_texts, true_texts)
            
            elif metric == 'lcs':
                similarities = np.array([[
                    self.calculate_lcs_similarity(v1, v2)
                    for v2 in true_texts
                ] for v1 in new_texts])
                return similarities
            
            elif metric == 'euclidean':
                from scipy.spatial.distance import cdist
                distances = cdist(new_texts, true_texts, metric='euclidean')
                similarities = 1 / (1 + distances)
                return similarities
        
        # Text or mixed input similarity calculation
        similarities = np.zeros((len(new_texts), len(true_texts)))
        
        if metric == 'jaccard':
            # Extract Jaccard-specific parameters
            n = kwargs.get('n', 1)
            use_char_ngrams = kwargs.get('use_char_ngrams', True)
            use_word_ngrams = kwargs.get('use_word_ngrams', True)
            char_weight = kwargs.get('char_weight', 0.5)
            
            # Pre-compute n-grams for all texts
            if use_char_ngrams:
                new_char_ngrams = [self._generate_character_ngrams(str(text), n) for text in new_texts]
                true_char_ngrams = [self._generate_character_ngrams(str(text), n) for text in true_texts]
            
            if use_word_ngrams:
                new_word_ngrams = [self._generate_word_ngrams(str(text), n) for text in new_texts]
                true_word_ngrams = [self._generate_word_ngrams(str(text), n) for text in true_texts]
            
            # Compute similarities
            for i, new_text in enumerate(new_texts):
                for j, true_text in enumerate(true_texts):
                    similarities[i, j] = self.calculate_jaccard_similarity(
                        new_text, true_text, n=n,
                        use_char_ngrams=use_char_ngrams,
                        use_word_ngrams=use_word_ngrams,
                        char_weight=char_weight
                    )
        
        elif metric in ['euclidean', 'levenshtein']:
            for i, new_text in enumerate(new_texts):
                for j, true_text in enumerate(true_texts):
                    if metric == 'euclidean':
                        similarities[i, j] = self.calculate_euclidean_similarity(new_text, true_text)
                    else:  # levenshtein
                        similarities[i, j] = self.calculate_levenshtein_similarity(new_text, true_text)
        
        else:
            self.logger.warning(f"Metric {metric} not supported for mixed/text inputs. Returning zero matrix.")
        
        return similarities

    def get_true_embeddings_with_ids(self) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve TRUE set embeddings and their corresponding text IDs
        """
        try:
            query = "SELECT embedding, text_id FROM embeddings WHERE is_true = True"
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
            query = "SELECT embedding, text_id FROM embeddings WHERE is_true = False"
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