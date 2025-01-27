# src/embeddings/generator.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
import logging
from config import config

class EmbeddingGenerator:
   def __init__(self, config: Dict):
       """
       Initialize embedding generator
       Args:
           config: Configuration with model settings
       """
       self.model_name = config.get('model_name', 'all-MiniLM-L6-v2')
       self.model = SentenceTransformer(self.model_name)
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       self.model.to(self.device)
       self.embedding_size = self.model.get_sentence_embedding_dimension()
       self.logger = logging.getLogger(__name__)

   def generate_embedding(self, text: str) -> np.ndarray:
       """
       Generate single text embedding
       Args:
           text: Input text
       Returns:
           Embedding vector
       """
       try:
           embedding = self.model.encode(text)
           return embedding
       except Exception as e:
           self.logger.error(f"Error generating embedding: {str(e)}")
           raise

   def batch_generate(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
       """
       Generate embeddings for text batch
       Args:
           texts: List of input texts
           batch_size: Batch size for processing
       Returns:
           Matrix of embeddings
       """
       embeddings = []
       for i in range(0, len(texts), batch_size):
           batch = texts[i:i + batch_size]
           try:
               batch_embeddings = self.model.encode(batch)
               embeddings.extend(batch_embeddings)
           except Exception as e:
               self.logger.error(f"Error in batch {i}: {str(e)}")
               continue
       return np.array(embeddings)

   def save_embeddings(self, embeddings: np.ndarray, filepath: Path):
       """
       Save embeddings to file
       Args:
           embeddings: Embedding matrix
           filepath: Output path
       """
       try:
           np.save(filepath, embeddings)
           self.logger.info(f"Saved embeddings to {filepath}")
       except Exception as e:
           self.logger.error(f"Error saving embeddings: {str(e)}")
           raise

if __name__ == "__main__":
   # Test the generator
   from src.preprocessing.loader import DataLoader
   from src.preprocessing.cleaner import TextPreprocessor

   # Initialize components
   loader = DataLoader()
   cleaner = TextPreprocessor(config)
   generator = EmbeddingGenerator(config)

   # Load and process texts
   true_df = loader.load_true_set()
   cleaned_true = cleaner.batch_process(true_df['texts'].tolist())

   # Generate embeddings
   embeddings = generator.batch_generate(cleaned_true)
   print(f"Generated embeddings shape: {embeddings.shape}")
   print(f"Embedding dimension: {generator.embedding_size}")