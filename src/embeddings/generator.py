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
from src.storage.db_manager import DatabaseManager

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
        self.db_manager = DatabaseManager(config)

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

    def batch_generate(self, texts: List[str], 
                    lens_ids: List[str], 
                    is_true_set: bool = False, 
                    batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_ids = lens_ids[i:i + batch_size]
            try:
                # Generate embeddings for the batch
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
                
                # Prepare metadata for database storage
                metadata = {
                    'ids': batch_ids,
                    'is_true': [is_true_set] * len(batch)
                }
                
                # Store embeddings in database
                self.db_manager.store_embeddings(
                    np.array(batch_embeddings), 
                    metadata
                )
            except Exception as e:
                self.logger.error(f"Error in batch {i}: {str(e)}")
                continue
        
        return np.array(embeddings)

if __name__ == "__main__":
    # Test the generator
    from src.preprocessing.loader import DataLoader
    from src.preprocessing.cleaner import TextPreprocessor
    
    # Initialize components
    loader = DataLoader()
    cleaner = TextPreprocessor(config)
    generator = EmbeddingGenerator(config)
    
    # Load and process TRUE set texts
    true_df = loader.load_true_set()
    cleaned_true = cleaner.batch_process(true_df['texts'].tolist())
    
    # Generate and store TRUE set embeddings
    true_embeddings = generator.batch_generate(
        cleaned_true, 
        true_df['lens_id'].tolist(),
        is_true_set=True
    )
    print(f"Generated TRUE set embeddings shape: {true_embeddings.shape}")
    
    # Load and process new texts
    new_df = loader.load_new_texts()
    cleaned_new = cleaner.batch_process(new_df['texts'].tolist())
    
    # Generate and store new texts embeddings
    new_embeddings = generator.batch_generate(
        cleaned_new, 
        new_df['lens_id'].tolist(),
        is_true_set=False
    )
    print(f"Generated new texts embeddings shape: {new_embeddings.shape}")