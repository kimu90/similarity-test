# src/analysis/classifier.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
import logging
from config import config

@dataclass
class ClassificationResult:
   label: bool
   confidence: float
   similarity_score: float
   lens_id: str

class TextClassifier:
   def __init__(self, config: Dict):
       """
       Initialize classifier
       Args:
           config: Configuration with threshold settings
       """
       self.threshold = config.get('threshold', 0.8)
       self.min_confidence = config.get('min_confidence', 0.6)
       self.logger = logging.getLogger(__name__)

   def classify_text(self, similarity: float, lens_id: str) -> ClassificationResult:
       """
       Classify single text based on similarity
       Args:
           similarity: Similarity score
           lens_id: Text identifier
       Returns:
           Classification result
       """
       is_similar = similarity >= self.threshold
       confidence = self._calculate_confidence(similarity)
       
       return ClassificationResult(
           label=is_similar,
           confidence=confidence,
           similarity_score=similarity,
           lens_id=lens_id
       )

   def batch_classify(self, similarities: np.ndarray, 
                     lens_ids: List[str]) -> List[ClassificationResult]:
       """
       Classify batch of texts
       Args:
           similarities: Array of similarity scores
           lens_ids: List of text identifiers
       Returns:
           List of classification results
       """
       return [
           self.classify_text(score, id)
           for score, id in zip(similarities, lens_ids)
       ]

   def _calculate_confidence(self, similarity: float) -> float:
       """
       Calculate classification confidence
       Args:
           similarity: Similarity score
       Returns:
           Confidence score
       """
       if similarity >= self.threshold:
           return min(1.0, similarity / self.threshold)
       return min(1.0, (1 - similarity) / (1 - self.threshold))

   def evaluate(self, predictions: List[bool], 
               ground_truth: List[bool]) -> Dict[str, float]:
       """
       Calculate classification metrics
       Args:
           predictions: Predicted labels
           ground_truth: True labels
       Returns:
           Dictionary of metrics
       """
       precision, recall, f1, _ = precision_recall_fscore_support(
           ground_truth, predictions, average='binary'
       )
       return {
           'precision': precision,
           'recall': recall,
           'f1': f1
       }

if __name__ == "__main__":
   # Test classifier
   from src.preprocessing.loader import DataLoader
   from src.preprocessing.cleaner import TextPreprocessor
   from src.embeddings.generator import EmbeddingGenerator
   from src.analysis.similarity import SimilarityAnalyzer

   # Initialize components
   loader = DataLoader()
   cleaner = TextPreprocessor(config)
   generator = EmbeddingGenerator(config)
   similarity = SimilarityAnalyzer(config)
   classifier = TextClassifier(config)

   # Load and process data
   true_df = loader.load_true_set()
   new_df = loader.load_new_texts(batch_size=100)

   cleaned_true = cleaner.batch_process(true_df['texts'].tolist())
   cleaned_new = cleaner.batch_process(new_df['texts'].tolist())

   # Generate embeddings
   true_embeddings = generator.batch_generate(cleaned_true)
   new_embeddings = generator.batch_generate(cleaned_new)

   # Calculate similarities and classify
   similarities = similarity.batch_similarities(new_embeddings, true_embeddings)
   classifications = classifier.batch_classify(
       similarities.max(axis=1), 
       new_df['lens_id'].tolist()
   )

   # Print results
   print("\nClassification Results:")
   for result in classifications[:5]:  # Show first 5 results
       print(f"\nID: {result.lens_id}")
       print(f"Label: {'Similar' if result.label else 'Different'}")
       print(f"Confidence: {result.confidence:.3f}")
       print(f"Similarity Score: {result.similarity_score:.3f}")