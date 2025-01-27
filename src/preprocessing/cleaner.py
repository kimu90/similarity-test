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

if __name__ == "__main__":
    # Test the cleaner
    from loader import DataLoader
    
    loader = DataLoader()
    cleaner = TextPreprocessor(config)
    
    # Load data
    true_df = loader.load_true_set()
    new_df = loader.load_new_texts()
    
    # Clean texts
    cleaned_true = cleaner.batch_process(true_df['texts'].tolist())
    cleaned_new = cleaner.batch_process(new_df['texts'].tolist())
    
    print(f"Processed {len(cleaned_true)} TRUE texts")
    print(f"Processed {len(cleaned_new)} new texts")