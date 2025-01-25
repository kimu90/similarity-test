# web/dashboard/controls.py

from typing import Dict, Optional
from dataclasses import dataclass
import logging

@dataclass
class FilterSettings:
   min_similarity: float
   max_similarity: float
   min_confidence: float
   date_range: Optional[tuple]
   label: Optional[bool]

class DashboardControls:
   def __init__(self, config: Dict):
       self.current_filters = FilterSettings(
           min_similarity=0.0,
           max_similarity=1.0,
           min_confidence=0.0,
           date_range=None,
           label=None
       )
       self.threshold = config.get('initial_threshold', 0.8)
       self.logger = logging.getLogger(__name__)

   def update_filters(self, filter_params: Dict) -> FilterSettings:
       """Update filter settings"""
       self.current_filters = FilterSettings(
           min_similarity=filter_params.get('min_similarity', self.current_filters.min_similarity),
           max_similarity=filter_params.get('max_similarity', self.current_filters.max_similarity),
           min_confidence=filter_params.get('min_confidence', self.current_filters.min_confidence),
           date_range=filter_params.get('date_range', self.current_filters.date_range),
           label=filter_params.get('label', self.current_filters.label)
       )
       return self.current_filters

   def apply_threshold(self, threshold: float) -> bool:
       """Update classification threshold"""
       if 0 <= threshold <= 1:
           self.threshold = threshold
           return True
       return False

   def get_query_params(self) -> Dict:
       """Convert current filters to query parameters"""
       params = {
           'min_similarity': self.current_filters.min_similarity,
           'max_similarity': self.current_filters.max_similarity,
           'min_confidence': self.current_filters.min_confidence
       }
       if self.current_filters.date_range:
           params['start_date'], params['end_date'] = self.current_filters.date_range
       if self.current_filters.label is not None:
           params['label'] = self.current_filters.label
       return params