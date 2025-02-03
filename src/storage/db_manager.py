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
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800
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
        """Initialize database tables"""
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
            
            # Define classifications table with unique constraint on text_id and metric
            sa.Table(
                'classifications', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('text_id', sa.String),
                sa.Column('similarity_score', sa.Float),
                sa.Column('metric', sa.String),
                sa.Column('confidence', sa.Float),
                sa.Column('label', sa.Boolean),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
                sa.UniqueConstraint('text_id', 'metric', name='uix_text_metric')  # Add this line
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
        Store classification results for a single metric
        
        Args:
            results: List of classification result dictionaries
        """
        session = self.SessionLocal()
        try:
            # Prepare data for bulk insert
            insert_data = [
                {
                    'text_id': r['text_id'],
                    'similarity_score': float(r['similarity_score']),
                    'metric': r['metric'],
                    'confidence': float(r['confidence']),
                    'label': bool(r['label'])
                }
                for r in results
            ]
            
            # Bulk insert/update using ON CONFLICT
            session.execute(
                sa.text(
                    """
                    INSERT INTO classifications 
                    (text_id, similarity_score, metric, confidence, label) 
                    VALUES (:text_id, :similarity_score, :metric, :confidence, :label)
                    ON CONFLICT (text_id, metric) DO UPDATE 
                    SET similarity_score = EXCLUDED.similarity_score,
                        confidence = EXCLUDED.confidence,
                        label = EXCLUDED.label
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
                       max_score: float = 1.0,
                       metric: str = 'cosine') -> pd.DataFrame:
        """
        Query classifications by similarity score for a specific metric
        """
        try:
            # Note that we're querying 'similarity_score' not 'cosine_score'
            query = """
                SELECT 
                    text_id,
                    similarity_score,  
                    metric,
                    confidence,
                    label,
                    created_at
                FROM classifications
                WHERE metric = :metric
                AND similarity_score BETWEEN :min_score AND :max_score
                ORDER BY similarity_score DESC
            """
            
            with self.engine.connect() as connection:
                df = pd.read_sql(
                    sa.text(query), 
                    connection, 
                    params={
                        'metric': metric,
                        'min_score': min_score, 
                        'max_score': max_score
                    }
                )
                
                # If needed, rename the column for compatibility
                if 'similarity_score' in df.columns:
                    df[f'{metric}_score'] = df['similarity_score']
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error querying similarities: {str(e)}")
            # Return empty DataFrame with correct column names
            return pd.DataFrame(columns=[
                'text_id', 
                'similarity_score', 
                f'{metric}_score',  # Add both column names for compatibility
                'metric', 
                'confidence', 
                'label'
            ])

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
        """Close database engine"""
        try:
            self.engine.dispose()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}")

if __name__ == "__main__":
    # Test database manager
    from config import config
    
    # Initialize database manager
    db_manager = DatabaseManager(config)
    
    # Test embedding storage
    test_embeddings = np.random.rand(5, 10)
    test_metadata = {
        'ids': [f'test_{i}' for i in range(5)],
        'is_true': [True] * 5
    }
    
    # Store test embeddings
    db_manager.store_embeddings(test_embeddings, test_metadata)
    
    # Test classification storage
    test_results = [
        {
            'text_id': f'test_{i}',
            'similarity_score': float(np.random.rand()),
            'metric': 'cosine',
            'confidence': float(np.random.rand()),
            'label': bool(np.random.choice([True, False]))
        }
        for i in range(5)
    ]
    
    # Store test results
    db_manager.store_results(test_results)
    
    # Test querying
    results = db_manager.query_by_similarity(metric='cosine')
    print("\nQuery Results:")
    print(results)
    
    # Close connection
    db_manager.close_connection()