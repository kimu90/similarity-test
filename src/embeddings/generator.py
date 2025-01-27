# src/embeddings/generator.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Optional
import torch
from sentence_transformers import SentenceTransformer
import logging
from config import config

class EmbeddingGenerator:
    def __init__(self, config: Dict, db_manager=None):
        """
        Initialize embedding generator
        
        Args:
            config: Configuration with model settings
            db_manager: Optional database manager
        """
        # Model configuration
        self.model_name = config.get('model_name', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(self.model_name)
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Embedding details
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Lazy import of DatabaseManager to avoid circular imports
        if db_manager is None:
            from src.storage.db_manager import DatabaseManager
            db_manager = DatabaseManager(config)
        
        self.db_manager = db_manager

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

    def batch_generate(self, 
                   texts: List[str],
                   lens_ids: List[str],
                   is_true_set: bool = False,
                   batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
            lens_ids: Corresponding list of text identifiers
            is_true_set: Flag to indicate if these are TRUE set texts
            batch_size: Number of texts to process in each batch
        
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            self.logger.error("No texts provided for embedding generation")
            return np.array([])  # Return empty array instead of raising an error

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
                
                self.logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} texts")
            
            except Exception as e:
                self.logger.error(f"Error processing batch {i}: {str(e)}")
                continue
        
        # Convert to numpy array, handling empty case
        embeddings_array = np.array(embeddings)
        
        if embeddings_array.size == 0:
            self.logger.error("No embeddings were generated")
        
        return embeddings_array

if __name__ == "__main__":
    # Test the generator
    from src.preprocessing.loader import DataLoader
    from src.preprocessing.cleaner import TextPreprocessor
    from src.storage.db_manager import DatabaseManager

    # Initialize components
    db_manager = DatabaseManager(config)
    loader = DataLoader()
    cleaner = TextPreprocessor(config)
    generator = EmbeddingGenerator(config, db_manager)

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