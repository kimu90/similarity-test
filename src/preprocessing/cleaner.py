# src/preprocessing/cleaner.py

import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

class TextPreprocessor:
   def __init__(self, config: Dict):
       """Initialize with preprocessing config settings"""
       self.remove_stops = config.get('remove_stopwords', True)
       self.min_length = config.get('min_length', 10)
       self.logger = logging.getLogger(__name__)
       nltk.download('punkt')
       nltk.download('stopwords')
       self.stops = set(stopwords.words('english'))

   def clean_text(self, text: str) -> str:
       """
       Main cleaning function
       Args:
           text: Raw text string
       Returns: Cleaned text
       """
       text = self.remove_noise(text)
       text = self.normalize_text(text)
       if self.remove_stops:
           text = self.remove_stopwords(text)
       return text

   def remove_noise(self, text: str) -> str:
       """Remove special chars, HTML, extra whitespace"""
       # Remove HTML
       text = re.sub('<[^<]+?>', '', text)
       # Remove special chars
       text = re.sub('[^A-Za-z0-9\s]', '', text)
       # Remove extra whitespace
       text = ' '.join(text.split())
       return text.strip()

   def normalize_text(self, text: str) -> str:
       """Normalize case, numbers, etc"""
       text = text.lower()
       # Convert numbers to words
       text = re.sub(r'\d+', 'NUM', text)
       return text

   def remove_stopwords(self, text: str) -> str:
       """Remove common stopwords"""
       tokens = word_tokenize(text)
       tokens = [t for t in tokens if t not in self.stops]
       return ' '.join(tokens)

   def batch_process(self, texts: List[str]) -> List[str]:
       """Process a batch of texts"""
       cleaned = []
       for text in texts:
           try:
               clean = self.clean_text(text)
               if len(clean.split()) >= self.min_length:
                   cleaned.append(clean)
           except Exception as e:
               self.logger.error(f"Error cleaning text: {str(e)}")
               continue
       return cleaned