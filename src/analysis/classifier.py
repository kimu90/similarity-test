# src/analysis/classifier.py

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
import logging

@dataclass
class ClassificationResult:
   label: bool
   confidence: float
   similarity_score: float
   text_id: str

class TextClassifier:
   def __init__(self, config: Dict):
       self.threshold = config.get('threshold', 0.8)
       self.min_confidence = config.get('min_confidence', 0.6)
       self.logger = logging.getLogger(__name__)

   def classify_text(self, similarity: float, text_id: str) -> ClassificationResult:
       """Classify single text based on similarity"""
       is_similar = similarity >= self.threshold
       confidence = self._calculate_confidence(similarity)
       
       return ClassificationResult(
           label=is_similar,
           confidence=confidence,
           similarity_score=similarity,
           text_id=text_id
       )

   def batch_classify(self, similarities: np.ndarray, 
                     text_ids: List[str]) -> List[ClassificationResult]:
       """Classify batch of texts"""
       return [
           self.classify_text(score, id)
           for score, id in zip(similarities, text_ids)
       ]

   def _calculate_confidence(self, similarity: float) -> float:
       """Calculate classification confidence"""
       if similarity >= self.threshold:
           return min(1.0, similarity / self.threshold)
       return min(1.0, (1 - similarity) / (1 - self.threshold))

   def evaluate(self, predictions: List[bool], 
               ground_truth: List[bool]) -> Dict[str, float]:
       """Calculate classification metrics"""
       precision, recall, f1, _ = precision_recall_fscore_support(
           ground_truth, predictions, average='binary'
       )
       return {
           'precision': precision,
           'recall': recall,
           'f1': f1
       }