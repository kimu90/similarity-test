# src/preprocessing/cleaner.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import re
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
from config import config

class TextPreprocessor:
    def __init__(self, config: Dict):
        """Initialize with preprocessing config settings"""
        self.remove_stops = config.get('remove_stopwords', True)
        self.min_length = config.get('min_length', 5)  # Reduced from 10 to 5
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure NLTK resources are downloaded
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            self.logger.warning(f"Error downloading NLTK resources: {str(e)}")
        
        # Initialize stopwords
        self.stops = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """
        Main cleaning function
        
        Args:
            text: Raw text string
        
        Returns: 
            Cleaned text
        """
        # Check if text is valid
        if not isinstance(text, str):
            self.logger.warning(f"Invalid text type: {type(text)}")
            return ""
        
        try:
            # Remove noise
            text = self.remove_noise(text)
            
            # Normalize text
            text = self.normalize_text(text)
            
            # Remove stopwords if configured
            if self.remove_stops:
                text = self.remove_stopwords(text)
            
            return text
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            return ""

    def batch_process(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts
        
        Args:
            texts: List of input texts
        
        Returns:
            List of cleaned texts
        """
        # Log input
        self.logger.info(f"Starting batch processing. Input texts: {len(texts)}")
        
        # Validate input
        if not texts:
            self.logger.warning("Empty text list provided")
            return []
        
        # Clean texts
        cleaned = []
        for text in texts:
            try:
                clean = self.clean_text(text)
                
                # Apply length filter
                # Modify this to be less strict
                if len(clean.split()) >= self.min_length:
                    cleaned.append(clean)
                else:
                    self.logger.info(f"Text too short after cleaning (length {len(clean.split())}): {clean[:100]}...")
            except Exception as e:
                self.logger.error(f"Error processing text: {str(e)}")
                continue
        
        # Log results
        self.logger.info(f"Batch processing complete. Cleaned texts: {len(cleaned)}")
        
        return cleaned

    def remove_noise(self, text: str) -> str:
        """Remove special chars, HTML, extra whitespace"""
        # Remove HTML
        text = re.sub('<[^<]+?>', '', text)
        
        # Remove special chars, but keep some punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def normalize_text(self, text: str) -> str:
        """Normalize case, numbers, etc"""
        text = text.lower()
        
        # Convert numbers to a generic token instead of removing
        text = re.sub(r'\d+', 'NUM', text)
        
        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove common stopwords"""
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t.lower() not in self.stops]
            
            return ' '.join(tokens)
        except Exception as e:
            self.logger.error(f"Error removing stopwords: {str(e)}")
            return text