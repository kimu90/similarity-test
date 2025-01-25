# src/preprocessing/loader.py

from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
import logging

class DataLoader:
   """
   Loads and validates text data from TRUE set and new texts
   """
   def __init__(self, config: Dict):
       self.true_set_path = Path(config['true_set_path'])
       self.new_texts_path = Path(config['new_texts_path']) 
       self.logger = logging.getLogger(__name__)

   def load_true_set(self) -> List[str]:
       """
       Loads TRUE set texts
       Returns: List of texts from TRUE set
       Raises: FileNotFoundError if path invalid
       """
       texts = []
       try:
           # Load based on file type (txt, csv etc)
           if self.true_set_path.suffix == '.txt':
               with open(self.true_set_path) as f:
                   texts = f.readlines()
           elif self.true_set_path.suffix == '.csv':
               df = pd.read_csv(self.true_set_path)
               texts = df['text'].tolist()
               
           self.logger.info(f"Loaded {len(texts)} TRUE texts")
           return texts

       except Exception as e:
           self.logger.error(f"Error loading TRUE set: {str(e)}")
           raise

   def load_new_texts(self, batch_size: Optional[int] = None) -> List[str]:
       """
       Loads new texts to classify
       Args:
           batch_size: Optional batch size for processing
       Returns: List of new texts
       """
       texts = []
       try:
           if self.new_texts_path.suffix == '.txt':
               with open(self.new_texts_path) as f:
                   texts = f.readlines()
           elif self.new_texts_path.suffix == '.csv':
               df = pd.read_csv(self.new_texts_path)
               texts = df['text'].tolist()

           if batch_size:
               texts = texts[:batch_size]

           self.logger.info(f"Loaded {len(texts)} new texts")
           return texts

       except Exception as e:
           self.logger.error(f"Error loading new texts: {str(e)}")
           raise
           
   def validate_texts(self, texts: List[str]) -> bool:
       """
       Validates text data format and content
       Args:
           texts: List of texts to validate
       Returns: True if valid, False otherwise
       """
       if not texts:
           self.logger.error("Empty text list")
           return False

       # Add validation logic (length, format etc)
       valid = all(isinstance(t, str) and len(t.strip()) > 0 for t in texts)
       
       if not valid:
           self.logger.error("Invalid text format found")
           return False
           
       return True