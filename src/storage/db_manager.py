# src/storage/db_manager.py
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
import logging

class DatabaseManager:
    def __init__(self, config: Dict):
        """
        Initialize database connection and setup
        
        Args:
            config: Configuration dictionary with database connection details
        """
        try:
            self.conn = psycopg2.connect(**config['database'])
            self.conn.autocommit = True  # Enable autocommit
            self.logger = logging.getLogger(__name__)
            self._init_tables()
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise

    def _init_tables(self):
        """Initialize database tables with vector extension"""
        with self.conn.cursor() as cur:
            try:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create embeddings table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        text_id TEXT UNIQUE,
                        embedding vector(384),
                        is_true BOOLEAN,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS classifications (
                        id SERIAL PRIMARY KEY,
                        text_id TEXT,
                        similarity_score FLOAT,
                        confidence FLOAT,
                        label BOOLEAN,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for faster searches
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_is_true 
                    ON embeddings(is_true);
                """)
            except Exception as e:
                self.logger.error(f"Table initialization error: {str(e)}")
                raise

    def store_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, bool]):
        """
        Store embeddings in the database
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: Dictionary containing text ids and true/false labels
        """
        with self.conn.cursor() as cur:
            try:
                # Convert numpy array to list of lists
                embedding_lists = embeddings.tolist()
                
                # Prepare data for insertion
                data = [
                    (id, emb, is_true) 
                    for id, (emb, is_true) in zip(metadata['ids'], zip(embedding_lists, metadata['is_true']))
                ]
                
                # Use execute_values for bulk insert
                execute_values(
                    cur,
                    """
                    INSERT INTO embeddings (text_id, embedding, is_true) 
                    VALUES %s 
                    ON CONFLICT (text_id) DO UPDATE 
                    SET embedding = EXCLUDED.embedding, 
                        is_true = EXCLUDED.is_true
                    """,
                    data
                )
            except Exception as e:
                self.logger.error(f"Error storing embeddings: {str(e)}")
                raise

    def store_results(self, results: List[Dict]):
        """
        Store classification results
        
        Args:
            results: List of classification result dictionaries
        """
        with self.conn.cursor() as cur:
            try:
                execute_values(
                    cur,
                    """
                    INSERT INTO classifications 
                    (text_id, similarity_score, confidence, label) 
                    VALUES %s
                    """,
                    [
                        (r['text_id'], r['similarity'], r['confidence'], r['label'])
                        for r in results
                    ]
                )
            except Exception as e:
                self.logger.error(f"Error storing results: {str(e)}")
                raise

    def query_by_similarity(self, 
                             min_score: float = 0.0, 
                             max_score: float = 1.0) -> pd.DataFrame:
        """
        Query classifications by similarity score
        
        Args:
            min_score: Minimum similarity score
            max_score: Maximum similarity score
        
        Returns:
            DataFrame of classification results
        """
        query = """
            SELECT * FROM classifications
            WHERE similarity_score BETWEEN %s AND %s
            ORDER BY similarity_score DESC
        """
        return pd.read_sql(query, self.conn, params=[min_score, max_score])

    def get_true_embeddings(self) -> np.ndarray:
        """
        Retrieve TRUE set embeddings
        
        Returns:
            Numpy array of TRUE set embeddings
        """
        query = "SELECT embedding FROM embeddings WHERE is_true = True"
        df = pd.read_sql(query, self.conn)
        return np.array(df['embedding'].tolist())

    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()