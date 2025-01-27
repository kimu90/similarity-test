# src/storage/db_manager.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
import json

class DatabaseManager:
    def __init__(self, config: Dict):
        """
        Initialize database connection using SQLAlchemy
        
        Args:
            config: Configuration dictionary with database connection details
        """
        try:
            # Construct database URL
            db_url = (
                f"postgresql://{config['database']['user']}:"
                f"{config['database']['password']}@"
                f"{config['database']['host']}:"
                f"{config['database']['port']}/"
                f"{config['database']['dbname']}"
            )
            
            # Create SQLAlchemy engine
            self.engine = sa.create_engine(
                db_url, 
                pool_size=10,  # Connection pool size
                max_overflow=20,  # Max additional connections
                pool_timeout=30,  # Timeout for getting a connection from pool
                pool_recycle=1800  # Recycle connections after 30 minutes
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Setup logging
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Initialize tables
            self._init_tables()
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise

    def _init_tables(self):
        """
        Initialize database tables with vector extension
        Uses SQLAlchemy metadata for table creation
        """
        try:
            # Create metadata object
            metadata = sa.MetaData()
            
            # Define embeddings table
            sa.Table(
                'embeddings', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('text_id', sa.String, unique=True),
                sa.Column('embedding', sa.JSON),  # Store as JSON
                sa.Column('is_true', sa.Boolean),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
            )
            
            # Define classifications table
            sa.Table(
                'classifications', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('text_id', sa.String),
                sa.Column('similarity_score', sa.Float),
                sa.Column('confidence', sa.Float),
                sa.Column('label', sa.Boolean),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
            )
            
            # Create all tables
            metadata.create_all(self.engine)
            
            self.logger.info("Database tables initialized successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error initializing tables: {str(e)}")
            raise

    def store_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, bool]):
        """
        Store embeddings in the database
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: Dictionary containing text ids and true/false labels
        """
        session = self.SessionLocal()
        try:
            # Convert embeddings to list of JSON-serializable lists
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Prepare data for bulk insert
            insert_data = [
                {
                    'text_id': id,
                    'embedding': json.dumps(emb),
                    'is_true': is_true
                }
                for id, (emb, is_true) in zip(metadata['ids'], zip(embedding_lists, metadata['is_true']))
            ]
            
            # Bulk insert with SQLAlchemy
            session.execute(
                sa.text(
                    """
                    INSERT INTO embeddings (text_id, embedding, is_true) 
                    VALUES (:text_id, :embedding, :is_true)
                    ON CONFLICT (text_id) DO UPDATE 
                    SET embedding = EXCLUDED.embedding, 
                        is_true = EXCLUDED.is_true
                    """
                ),
                insert_data
            )
            
            session.commit()
            self.logger.info(f"Stored {len(insert_data)} embeddings")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing embeddings: {str(e)}")
            raise
        finally:
            session.close()

    def store_results(self, results: List[Dict]):
        """
        Store classification results
        
        Args:
            results: List of classification result dictionaries
        """
        session = self.SessionLocal()
        try:
            # Prepare data for bulk insert, converting numpy types
            insert_data = [
                {
                    'text_id': r['text_id'],
                    'similarity_score': float(r['similarity']),  # Convert to Python float
                    'confidence': float(r['confidence']),  # Convert to Python float
                    'label': bool(r['label'])  # Ensure boolean
                }
                for r in results
            ]
            
            # Bulk insert
            session.execute(
                sa.text(
                    """
                    INSERT INTO classifications 
                    (text_id, similarity_score, confidence, label) 
                    VALUES (:text_id, :similarity_score, :confidence, :label)
                    """
                ),
                insert_data
            )
            
            session.commit()
            self.logger.info(f"Stored {len(insert_data)} classification results")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing results: {str(e)}")
            raise
        finally:
            session.close()

    def query_by_similarity(self, 
                         min_score: float = 0.0, 
                         max_score: float = 1.0) -> pd.DataFrame:
        """
        Query classifications by similarity score
        """
        try:
            # Use a more robust query method
            query = """
                SELECT * FROM classifications
                WHERE similarity_score BETWEEN :min_score AND :max_score
                ORDER BY similarity_score DESC
            """
            
            # Use SQLAlchemy engine connection
            with self.engine.connect() as connection:
                return pd.read_sql(
                    sa.text(query), 
                    connection, 
                    params={
                        'min_score': min_score, 
                        'max_score': max_score
                    }
                )
        except Exception as e:
            self.logger.error(f"Error querying similarities: {str(e)}")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=['text_id', 'similarity_score', 'confidence', 'label'])

    def get_true_embeddings(self) -> np.ndarray:
        """
        Retrieve TRUE set embeddings
        
        Returns:
            Numpy array of TRUE set embeddings
        """
        try:
            query = "SELECT embedding FROM embeddings WHERE is_true = True"
            df = pd.read_sql(query, self.engine)
            
            # Parse JSON and convert to numpy array
            return np.array([
                np.array(json.loads(emb)) for emb in df['embedding']
            ])
        except Exception as e:
            self.logger.error(f"Error retrieving TRUE embeddings: {str(e)}")
            raise

    def close_connection(self):
        """
        Close database engine
        """
        try:
            self.engine.dispose()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}")