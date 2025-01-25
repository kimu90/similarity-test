# src/embeddings/generator.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import torch
import logging
from pathlib import Path

class EmbeddingGenerator:
   def __init__(self, config: Dict):
       self.model_name = config.get('model_name', 'all-MiniLM-L6-v2')
       self.model = SentenceTransformer(self.model_name)
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       self.model.to(self.device)
       self.embedding_size = self.model.get_sentence_embedding_dimension()
       self.logger = logging.getLogger(__name__)

   def generate_embedding(self, text: str) -> np.ndarray:
       """Generate single text embedding"""
       try:
           embedding = self.model.encode(text)
           return embedding
       except Exception as e:
           self.logger.error(f"Error generating embedding: {str(e)}")
           raise

   def batch_generate(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
       """Generate embeddings for text batch"""
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
       """Save embeddings to file"""
       try:
           np.save(filepath, embeddings)
           self.logger.info(f"Saved embeddings to {filepath}")
       except Exception as e:
           self.logger.error(f"Error saving embeddings: {str(e)}")
           raise