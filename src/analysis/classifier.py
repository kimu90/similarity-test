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
    metric: str

class TextClassifier:
    def __init__(self, config: Dict, db_manager=None):
        """
        Initialize classifier with configuration
        
        Args:
            config: Configuration dictionary
            db_manager: Optional database manager
        """
        # Threshold settings
        self.thresholds = {
            'cosine': config.get('cosine_threshold', 0.8),
            'jaccard': config.get('jaccard_threshold', 0.8),
            'lcs': config.get('lcs_threshold', 0.8)
        }
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        # Lazy import of DatabaseManager to avoid circular imports
        if db_manager is None:
            from src.storage.db_manager import DatabaseManager
            db_manager = DatabaseManager(config)
        
        self.db_manager = db_manager

    def classify_text(self, 
                     similarity_score: float, 
                     text_id: str,
                     metric: str = 'cosine') -> ClassificationResult:
        """
        Classify a single text based on similarity score
        
        Args:
            similarity_score: Similarity score
            text_id: Unique identifier for the text
            metric: Similarity metric used
        
        Returns:
            ClassificationResult object
        """
        threshold = self.thresholds[metric]
        is_similar = similarity_score >= threshold
        confidence = self._calculate_confidence(similarity_score, threshold)
        
        return ClassificationResult(
            label=is_similar,
            confidence=confidence,
            similarity_score=similarity_score,
            text_id=text_id,
            metric=metric
        )

    def batch_classify(self, 
                      similarity_scores: np.ndarray,
                      text_ids: List[str],
                      metric: str = 'cosine') -> List[ClassificationResult]:
        """
        Classify a batch of texts
        
        Args:
            similarity_scores: Array of similarity scores
            text_ids: List of text identifiers
            metric: Similarity metric used
        
        Returns:
            List of ClassificationResult objects
        """
        return [
            self.classify_text(score, id, metric)
            for score, id in zip(similarity_scores, text_ids)
        ]

    def _calculate_confidence(self, similarity: float, threshold: float) -> float:
        """
        Calculate confidence score based on similarity
        
        Args:
            similarity: Similarity score
            threshold: Classification threshold
        
        Returns:
            Confidence score between 0 and 1
        """
        if similarity >= threshold:
            # Normalize confidence for scores above threshold
            return min(1.0, similarity / threshold)
        else:
            # Normalize confidence for scores below threshold
            return min(1.0, (1 - similarity) / (1 - threshold))

    def store_classifications(self, classifications: List[ClassificationResult]):
        """
        Store classification results in database
        
        Args:
            classifications: List of classification results
        """
        # Prepare results for database storage
        results = [
            {
                'text_id': result.text_id,
                'similarity_score': float(result.similarity_score),
                'metric': result.metric,
                'confidence': float(result.confidence),
                'label': bool(result.label)
            }
            for result in classifications
        ]
        
        # Store in database
        self.db_manager.store_results(results)

    def get_similarity_results(self, 
                             min_score: float = 0.0, 
                             max_score: float = 1.0,
                             metric: str = 'cosine') -> List[ClassificationResult]:
        """
        Retrieve similarity results from database
        
        Args:
            min_score: Minimum similarity score
            max_score: Maximum similarity score
            metric: Similarity metric to query
        
        Returns:
            List of ClassificationResult objects
        """
        try:
            query = """
                SELECT text_id, similarity_score, confidence, label 
                FROM classifications 
                WHERE metric = :metric
                AND similarity_score BETWEEN :min_score AND :max_score
            """
            df = pd.read_sql(
                query, 
                self.db_manager.engine, 
                params={
                    'metric': metric,
                    'min_score': min_score, 
                    'max_score': max_score
                }
            )
            
            # Convert to ClassificationResult objects
            return [
                ClassificationResult(
                    label=row['label'],
                    confidence=row['confidence'],
                    similarity_score=row['similarity_score'],
                    text_id=row['text_id'],
                    metric=metric
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

    def get_metric_threshold(self, metric: str) -> float:
        """
        Get threshold for specific metric
        
        Args:
            metric: Similarity metric
        
        Returns:
            Threshold value for the metric
        """
        return self.thresholds.get(metric, 0.8)

if __name__ == "__main__":
    # Test classifier
    from src.analysis.similarity import SimilarityAnalyzer
    from src.storage.db_manager import DatabaseManager

    # Initialize components
    db_manager = DatabaseManager(config)
    similarity = SimilarityAnalyzer(config, db_manager)
    classifier = TextClassifier(config, db_manager)

    # Test with some random data
    test_scores = np.random.rand(10)
    test_ids = [f"test_{i}" for i in range(10)]

    # Test classification for each metric
    for metric in ['cosine', 'jaccard', 'lcs']:
        print(f"\nTesting {metric} classification:")
        
        # Classify texts
        classifications = classifier.batch_classify(
            test_scores,
            test_ids,
            metric=metric
        )

        # Store classifications
        classifier.store_classifications(classifications)

        # Retrieve and print stored classifications
        stored_results = classifier.get_similarity_results(metric=metric)
        
        print("\nClassification Results:")
        for result in stored_results[:3]:  # Show first 3 results
            print(f"\nID: {result.text_id}")
            print(f"Label: {'Similar' if result.label else 'Different'}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Similarity Score: {result.similarity_score:.3f}")