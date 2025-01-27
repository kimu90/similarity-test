# web/dashboard/metrics.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import logging
from config import config

class MetricsCalculator:
   def __init__(self):
       """Initialize metrics calculator"""
       self.metrics_history = []
       self.logger = logging.getLogger(__name__)

   def calculate_metrics(self, predictions: List[bool], truth: List[bool]) -> Dict:
       """
       Calculate classification metrics
       Args:
           predictions: Predicted labels
           truth: True labels
       Returns:
           Dictionary of metrics
       """
       precision, recall, f1, _ = precision_recall_fscore_support(
           truth, predictions, average='binary'
       )
       
       return {
           'precision': precision,
           'recall': recall,
           'f1': f1
       }

   def calculate_roc(self, scores: np.ndarray, truth: List[bool]) -> Dict:
       """
       Calculate ROC curve data
       Args:
           scores: Similarity scores
           truth: True labels
       Returns:
           Dictionary with ROC curve data
       """
       fpr, tpr, thresholds = roc_curve(truth, scores)
       roc_auc = auc(fpr, tpr)
       
       return {
           'fpr': fpr.tolist(),
           'tpr': tpr.tolist(),
           'auc': roc_auc,
           'thresholds': thresholds.tolist()
       }

   def get_summary_stats(self, results: pd.DataFrame) -> Dict:
       """
       Calculate summary statistics
       Args:
           results: DataFrame with similarity results
       Returns:
           Dictionary of summary statistics
       """
       stats = {
           'total_processed': len(results),
           'true_ratio': results['label'].mean(),
           'avg_confidence': results['confidence'].mean(),
           'avg_similarity': results['similarity_score'].mean(),
           'std_similarity': results['similarity_score'].std(),
           'median_similarity': results['similarity_score'].median(),
           'threshold_violations': (
               results['confidence'] < results['similarity_score']
           ).sum()
       }
       
       # Add quartile information
       quartiles = results['similarity_score'].quantile([0.25, 0.5, 0.75])
       stats.update({
           'q1_similarity': quartiles[0.25],
           'q2_similarity': quartiles[0.50],
           'q3_similarity': quartiles[0.75]
       })
       
       return stats

   def track_metrics(self, new_metrics: Dict):
       """
       Track metrics over time
       Args:
           new_metrics: New metrics to track
       """
       self.metrics_history.append({
           'timestamp': pd.Timestamp.now(),
           **new_metrics
       })
       self.logger.info(f"Tracked new metrics: {new_metrics}")

   def get_metrics_trend(self) -> pd.DataFrame:
       """
       Get metrics history as DataFrame
       Returns:
           DataFrame with metrics history
       """
       return pd.DataFrame(self.metrics_history)

   def calculate_threshold_metrics(self, 
                                scores: np.ndarray, 
                                thresholds: List[float]) -> pd.DataFrame:
       """
       Calculate metrics for different thresholds
       Args:
           scores: Similarity scores
           thresholds: List of thresholds to test
       Returns:
           DataFrame with threshold metrics
       """
       results = []
       for threshold in thresholds:
           predictions = scores >= threshold
           metrics = self.calculate_metrics(predictions, [True] * len(scores))
           results.append({
               'threshold': threshold,
               **metrics
           })
       return pd.DataFrame(results)

if __name__ == "__main__":
   # Test metrics calculator
   from src.preprocessing.loader import DataLoader
   from src.analysis.similarity import SimilarityAnalyzer
   
   # Initialize components
   loader = DataLoader()
   similarity = SimilarityAnalyzer(config)
   metrics = MetricsCalculator()
   
   # Generate test data
   scores = np.random.rand(100)
   truth = [True if s > 0.7 else False for s in scores]
   
   # Calculate various metrics
   classification_metrics = metrics.calculate_metrics([s > 0.5 for s in scores], truth)
   roc_data = metrics.calculate_roc(scores, truth)
   
   # Test threshold analysis
   threshold_metrics = metrics.calculate_threshold_metrics(
       scores, 
       thresholds=[0.3, 0.5, 0.7, 0.9]
   )
   
   # Print results
   print("\nClassification Metrics:")
   print(classification_metrics)
   print("\nROC AUC:", roc_data['auc'])
   print("\nThreshold Analysis:")
   print(threshold_metrics)