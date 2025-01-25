# web/dashboard/metrics.py

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

class MetricsCalculator:
   def __init__(self):
       self.metrics_history = []

   def calculate_metrics(self, predictions: List[bool], truth: List[bool]) -> Dict:
       """Calculate classification metrics"""
       precision, recall, f1, _ = precision_recall_fscore_support(
           truth, predictions, average='binary'
       )
       
       return {
           'precision': precision,
           'recall': recall,
           'f1': f1
       }

   def calculate_roc(self, scores: np.ndarray, truth: List[bool]) -> Dict:
       """Calculate ROC curve data"""
       fpr, tpr, thresholds = roc_curve(truth, scores)
       roc_auc = auc(fpr, tpr)
       
       return {
           'fpr': fpr.tolist(),
           'tpr': tpr.tolist(),
           'auc': roc_auc,
           'thresholds': thresholds.tolist()
       }

   def get_summary_stats(self, results: pd.DataFrame) -> Dict:
       """Calculate summary statistics"""
       return {
           'total_processed': len(results),
           'true_ratio': results['label'].mean(),
           'avg_confidence': results['confidence'].mean(),
           'avg_similarity': results['similarity_score'].mean(),
           'threshold_violations': (
               results['confidence'] < results['similarity_score']
           ).sum()
       }

   def track_metrics(self, new_metrics: Dict):
       """Track metrics over time"""
       self.metrics_history.append({
           'timestamp': pd.Timestamp.now(),
           **new_metrics
       })

   def get_metrics_trend(self) -> pd.DataFrame:
       """Get metrics history as DataFrame"""
       return pd.DataFrame(self.metrics_history)