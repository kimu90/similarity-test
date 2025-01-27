# src/analysis/classifier.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import logging
import pandas as pd
from config import config

@dataclass
class ClassificationResult:
    """
    Dataclass to represent classification results
    """
    label: bool
    confidence: float
    similarity_score: float
    text_id: str

class TextClassifier:
    def __init__(self, config: Dict, db_manager=None):
        """
        Initialize classifier with configuration
        
        Args:
            config: Configuration dictionary
            db_manager: Optional database manager
        """
        # Threshold settings
        self.threshold = config.get('threshold', 0.8)
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        # Lazy import of DatabaseManager to avoid circular imports
        if db_manager is None:
            from src.storage.db_manager import DatabaseManager
            db_manager = DatabaseManager(config)
        
        self.db_manager = db_manager

    def classify_text(self, similarity: float, text_id: str) -> ClassificationResult:
        """
        Classify a single text based on similarity score
        
        Args:
            similarity: Similarity score
            text_id: Unique identifier for the text
        
        Returns:
            ClassificationResult object
        """
        is_similar = similarity >= self.threshold
        confidence = self._calculate_confidence(similarity)
        
        return ClassificationResult(
            label=is_similar,
            confidence=confidence,
            similarity_score=similarity,
            text_id=text_id
        )

    def batch_classify(self, 
                       similarities: np.ndarray, 
                       text_ids: List[str]) -> List[ClassificationResult]:
        """
        Classify a batch of texts
        
        Args:
            similarities: Array of similarity scores
            text_ids: List of text identifiers
        
        Returns:
            List of ClassificationResult objects
        """
        return [
            self.classify_text(score, id)
            for score, id in zip(similarities, text_ids)
        ]

    def _calculate_confidence(self, similarity: float) -> float:
        """
        Calculate confidence score based on similarity
        
        Args:
            similarity: Similarity score
        
        Returns:
            Confidence score between 0 and 1
        """
        if similarity >= self.threshold:
            # Normalize confidence for scores above threshold
            return min(1.0, similarity / self.threshold)
        else:
            # Normalize confidence for scores below threshold
            return min(1.0, (1 - similarity) / (1 - self.threshold))

    def store_classifications(self, classifications: List[ClassificationResult]):
        """
        Store classification results in database
        
        Args:
            classifications: List of classification results
        """
        try:
            # Prepare results for database storage
            results = [
                {
                    'text_id': result.text_id,
                    'similarity': result.similarity_score,
                    'confidence': result.confidence,
                    'label': result.label
                }
                for result in classifications
            ]
            
            # Store in database
            self.db_manager.store_results(results)
            
            self.logger.info(f"Stored {len(results)} classification results")
        
        except Exception as e:
            self.logger.error(f"Error storing classifications: {str(e)}")
            raise

    def get_similarity_results(self, 
                                min_score: float = 0.0, 
                                max_score: float = 1.0) -> List[ClassificationResult]:
        """
        Retrieve similarity results from database
        
        Args:
            min_score: Minimum similarity score
            max_score: Maximum similarity score
        
        Returns:
            List of ClassificationResult objects
        """
        try:
            query = """
                SELECT text_id, similarity_score, confidence, label 
                FROM classifications 
                WHERE similarity_score BETWEEN :min_score AND :max_score
            """
            df = pd.read_sql(
                query, 
                self.db_manager.engine, 
                params={'min_score': min_score, 'max_score': max_score}
            )
            
            # Convert to ClassificationResult objects
            return [
                ClassificationResult(
                    label=row['label'],
                    confidence=row['confidence'],
                    similarity_score=row['similarity_score'],
                    text_id=row['text_id']
                )
                for _, row in df.iterrows()
            ]
        
        except Exception as e:
            self.logger.error(f"Error retrieving similarity results: {str(e)}")
            return []

    def evaluate(self, 
                 predictions: List[bool], 
                 ground_truth: List[bool]) -> Dict[str, float]:
        """
        Calculate classification metrics
        
        Args:
            predictions: Predicted labels
            ground_truth: True labels
        
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Calculate precision, recall, and F1 score
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predictions, average='binary'
            )
            
            # Calculate confusion matrix
            cm = confusion_matrix(ground_truth, predictions)
            
            # Prepare metrics
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': cm[1, 1],
                'true_negatives': cm[0, 0],
                'false_positives': cm[0, 1],
                'false_negatives': cm[1, 0]
            }
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error evaluating classification: {str(e)}")
            return {}

if __name__ == "__main__":
    # Test classifier
    from src.analysis.similarity import SimilarityAnalyzer
    from src.storage.db_manager import DatabaseManager

    # Initialize components
    db_manager = DatabaseManager(config)
    similarity = SimilarityAnalyzer(config, db_manager)
    classifier = TextClassifier(config, db_manager)

    # Retrieve embeddings from database
    true_embeddings, true_text_ids = similarity.get_true_embeddings_with_ids()
    new_embeddings, new_text_ids = similarity.get_new_embeddings_with_ids()

    # Calculate similarities
    similarities = similarity.batch_similarities(new_embeddings, true_embeddings)

    # Classify texts
    classifications = classifier.batch_classify(
        similarities.max(axis=1), 
        new_text_ids
    )

    # Store classifications
    classifier.store_classifications(classifications)

    # Retrieve and print stored classifications
    stored_results = classifier.get_similarity_results()
    
    print("\nClassification Results:")
    for result in stored_results[:5]:  # Show first 5 results
        print(f"\nID: {result.text_id}")
        print(f"Label: {'Similar' if result.label else 'Different'}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Similarity Score: {result.similarity_score:.3f}")