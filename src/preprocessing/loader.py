# src/preprocessing/loader.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from config import config

class DataLoader:
    def __init__(self):
        """Initialize DataLoader with paths from config"""
        self.true_set_path = config['paths']['true_set_path']
        self.new_texts_path = config['paths']['new_texts_path']
        self.batch_size = config['data']['batch_size']
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def load_true_set(self) -> pd.DataFrame:
        """
        Load and validate TRUE set data
        Returns: DataFrame with TRUE set texts
        """
        try:
            self.logger.info(f"Loading TRUE set from {self.true_set_path}")
            df = pd.read_csv(self.true_set_path)
            
            if self.validate_data(df):
                self.logger.info(f"Successfully loaded {len(df)} TRUE set records")
                return df
            else:
                raise ValueError("TRUE set validation failed")
                
        except Exception as e:
            self.logger.error(f"Error loading TRUE set: {str(e)}")
            raise

    def load_new_texts(self, batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load new texts, optionally in batches
        Args:
            batch_size: Optional batch size override
        Returns: DataFrame with new texts
        """
        try:
            batch_size = batch_size or self.batch_size
            self.logger.info(f"Loading new texts from {self.new_texts_path}")
            
            if batch_size:
                # Use chunking for large files
                df = next(pd.read_csv(self.new_texts_path, chunksize=batch_size))
                self.logger.info(f"Loaded batch of {len(df)} new text records")
            else:
                df = pd.read_csv(self.new_texts_path)
                self.logger.info(f"Loaded {len(df)} new text records")
            
            if self.validate_data(df):
                return df
            else:
                raise ValueError("New texts validation failed")
                
        except Exception as e:
            self.logger.error(f"Error loading new texts: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame structure and content
        Args:
            df: DataFrame to validate
        Returns: True if valid, False otherwise
        """
        required_columns = {'lens_id', 'texts'}
        
        # Check columns
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Expected {required_columns}")
            return False
            
        # Check for empty texts
        if df['texts'].isna().any():
            self.logger.error("Found empty text entries")
            return False
            
        # Check lens_id uniqueness
        if df['lens_id'].duplicated().any():
            self.logger.error("Found duplicate lens_ids")
            return False
            
        # Validate text content
        if (df['texts'].str.len() < 10).any():
            self.logger.error("Found texts shorter than 10 characters")
            return False
            
        return True

    def get_data_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get basic statistics about the dataset
        Args:
            df: DataFrame to analyze
        Returns: Dictionary with statistics
        """
        return {
            'total_records': len(df),
            'avg_text_length': df['texts'].str.len().mean(),
            'unique_texts': df['texts'].nunique()
        }

if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    
    # Load both datasets
    true_df = loader.load_true_set()
    new_df = loader.load_new_texts(batch_size=100)
    
    # Print stats
    print("\nTRUE Set Stats:")
    print(loader.get_data_stats(true_df))
    print("\nNew Texts Stats:")
    print(loader.get_data_stats(new_df))