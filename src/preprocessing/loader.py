import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from config import config

# Import DatabaseManager from the second file
from src.storage.db_manager import DatabaseManager

class DataLoader:
    def __init__(self):
        self.true_set_path = config['paths']['true_set_path']
        self.new_texts_path = config['paths']['new_texts_path']
        self.batch_size = config['data']['batch_size']
        self.db = DatabaseManager(config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_total_rows(self) -> int:
        return sum(1 for _ in pd.read_csv(self.new_texts_path, chunksize=10000))

    def load_new_texts(self, batch_size: int, resume: bool, selected_metric: str):
        try:
            batch_size = batch_size or self.batch_size
            
            # Get processing status for the selected metric
            status = self.db.get_processing_status(metric=selected_metric)
            offset = status['offset'] if resume else 0
            
            if offset == 0 or not resume:
                total_rows = self.get_total_rows()
                self.db.update_processing_status(0, total_rows, 'in_progress', metric=selected_metric)
            
            self.logger.info(f"Loading new texts from {self.new_texts_path}, offset: {offset}")
            
            # Skip processed rows and load next batch
            df = pd.read_csv(
                self.new_texts_path,
                skiprows=range(1, offset + 1) if offset else None,
                nrows=batch_size
            )
            
            df = self.clean_column_names(df)
            
            if self.validate_data(df):
                # Update processing status for the selected metric
                new_offset = offset + len(df)
                self.db.update_processing_status(
                    new_offset,
                    status['total_rows'],
                    'completed' if new_offset >= status['total_rows'] else 'in_progress',
                    metric=selected_metric
                )
                
                self.logger.info(f"Loaded batch of {len(df)} records. Progress: {new_offset}/{status['total_rows']}")
                return df
            else:
                raise ValueError("New texts validation failed")
                
        except Exception as e:
            self.db.update_processing_status(offset, status['total_rows'], 'failed', metric=selected_metric)
            self.logger.error(f"Error loading new texts: {str(e)}")
            raise

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by stripping whitespace and handling the 'texts'/'text' case
        Args:
            df: DataFrame to clean
        Returns: DataFrame with cleaned column names
        """
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Handle 'texts' column if present
        if 'texts' in df.columns and 'text' not in df.columns:
            self.logger.info("Found 'texts' column, renaming to 'text'")
            df = df.rename(columns={'texts': 'text'})
            
        return df

    def load_true_set(self) -> pd.DataFrame:
        """
        Load and validate TRUE set data
        Returns: DataFrame with TRUE set texts
        """
        try:
            self.logger.info(f"Loading TRUE set from {self.true_set_path}")
            df = pd.read_csv(self.true_set_path)
            df = self.clean_column_names(df)
            
            if self.validate_data(df):
                self.logger.info(f"Successfully loaded {len(df)} TRUE set records")
                return df
            else:
                raise ValueError("TRUE set validation failed")
                
        except Exception as e:
            self.logger.error(f"Error loading TRUE set: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Basic check that required columns exist"""
        required_columns = {'lens_id', 'text'}
        return all(col in df.columns for col in required_columns)

    def get_data_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get basic statistics about the dataset
        Args:
            df: DataFrame to analyze
        Returns: Dictionary with statistics
        """
        return {
            'total_records': len(df),
            'avg_text_length': df['text'].str.len().mean(),
            'unique_texts': df['text'].nunique()
        }

if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    
    # Load both datasets
    true_df = loader.load_true_set()
    new_df = loader.load_new_texts(batch_size=10000)
    
    # Print stats
    print("\nTRUE Set Stats:")
    print(loader.get_data_stats(true_df))
    print("\nNew Texts Stats:")
    print(loader.get_data_stats(new_df))