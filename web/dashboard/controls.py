# web/dashboard/controls.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from config import config

@dataclass
class FilterSettings:
   """Settings for filtering similarity results"""
   min_similarity: float
   max_similarity: float
   min_confidence: float
   date_range: Optional[tuple]
   label: Optional[bool]

class DashboardControls:
   def __init__(self, config: Dict):
       """
       Initialize dashboard controls
       Args:
           config: Configuration dictionary
       """
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
       """
       Update filter settings
       Args:
           filter_params: New filter parameters
       Returns:
           Updated filter settings
       """
       try:
           self.current_filters = FilterSettings(
               min_similarity=filter_params.get('min_similarity', self.current_filters.min_similarity),
               max_similarity=filter_params.get('max_similarity', self.current_filters.max_similarity),
               min_confidence=filter_params.get('min_confidence', self.current_filters.min_confidence),
               date_range=filter_params.get('date_range', self.current_filters.date_range),
               label=filter_params.get('label', self.current_filters.label)
           )
           self.logger.info(f"Updated filters: {self.current_filters}")
           return self.current_filters
           
       except Exception as e:
           self.logger.error(f"Error updating filters: {str(e)}")
           raise

   def apply_threshold(self, threshold: float) -> bool:
       """
       Update classification threshold
       Args:
           threshold: New threshold value
       Returns:
           Success status
       """
       if 0 <= threshold <= 1:
           self.threshold = threshold
           self.logger.info(f"Updated threshold to {threshold}")
           return True
       else:
           self.logger.error(f"Invalid threshold value: {threshold}")
           return False

   def get_query_params(self) -> Dict:
       """
       Convert current filters to query parameters
       Returns:
           Dictionary of query parameters
       """
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

   def get_time_range_filters(self, range_type: str) -> tuple:
       """
       Get date range for filtering
       Args:
           range_type: Type of time range (day, week, month)
       Returns:
           Tuple of start and end dates
       """
       end_date = datetime.now()
       
       if range_type == 'day':
           start_date = end_date - timedelta(days=1)
       elif range_type == 'week':
           start_date = end_date - timedelta(weeks=1)
       elif range_type == 'month':
           start_date = end_date - timedelta(days=30)
       else:
           raise ValueError(f"Invalid time range type: {range_type}")
           
       return start_date, end_date

   def reset_filters(self):
       """Reset filters to default values"""
       self.current_filters = FilterSettings(
           min_similarity=0.0,
           max_similarity=1.0,
           min_confidence=0.0,
           date_range=None,
           label=None
       )
       self.logger.info("Reset filters to default values")

if __name__ == "__main__":
   # Test controls
   controls = DashboardControls(config)
   
   # Test filter updates
   new_filters = {
       'min_similarity': 0.7,
       'max_similarity': 0.9,
       'min_confidence': 0.6
   }
   
   # Update and print filters
   updated_filters = controls.update_filters(new_filters)
   print("\nUpdated Filters:")
   print(updated_filters)
   
   # Test threshold update
   threshold_success = controls.apply_threshold(0.85)
   print(f"\nThreshold update success: {threshold_success}")
   
   # Get query parameters
   query_params = controls.get_query_params()
   print("\nQuery Parameters:")
   print(query_params)
   
   # Test time range filters
   day_range = controls.get_time_range_filters('day')
   print("\nDay Range:")
   print(day_range)