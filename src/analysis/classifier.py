

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
        """
        self.thresholds = {
            'cosine': config.get('cosine_threshold', 0.8),
            'jaccard': config.get('jaccard_threshold', 0.8),
            'lcs': config.get('lcs_threshold', 0.8)
        }
        self.min_confidence = config.get('min_confidence', 0.6)
        
        self.logger = logging.getLogger(__name__)
        
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
        """
        return [
            self.classify_text(score, id, metric)
            for score, id in zip(similarity_scores, text_ids)
        ]

    def _calculate_confidence(self, similarity: float, threshold: float) -> float:
        """
        Calculate confidence score based on similarity
        """
        if similarity >= threshold:
            return min(1.0, similarity / threshold)
        else:
            return min(1.0, (1 - similarity) / (1 - threshold))

    def store_classifications(self, classifications: List[ClassificationResult]):
        """
        Store classification results in database
        """
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
        
        self.db_manager.store_results(results)

    def get_similarity_results(self, 
                             min_score: float = 0.0, 
                             max_score: float = 1.0,
                             metric: str = 'cosine') -> List[ClassificationResult]:
        """
        Retrieve similarity results from database
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
        """
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predictions, average='binary'
            )
            
            cm = confusion_matrix(ground_truth, predictions)
            
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
        """
        return self.thresholds.get(metric, 0.8)