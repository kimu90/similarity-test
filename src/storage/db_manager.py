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
from datetime import datetime

class DatabaseManager:
    def __init__(self, config: Dict):
        try:
            db_url = (
                f"postgresql://{config['database']['user']}:"
                f"{config['database']['password']}@"
                f"{config['database']['host']}:"
                f"{config['database']['port']}/"
                f"{config['database']['dbname']}"
            )
            
            self.engine = sa.create_engine(
                db_url, 
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800
            )
            
            self.SessionLocal = sessionmaker(bind=self.engine)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self._init_tables()
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise

    def _init_tables(self):
        try:
            metadata = sa.MetaData()
            
            # Updated embeddings table with batch tracking
            sa.Table(
                'embeddings', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('text_id', sa.String, unique=True),
                sa.Column('embedding', sa.JSON),
                sa.Column('is_true', sa.Boolean),
                sa.Column('batch_id', sa.String),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
            )
            
            # Updated classifications table with batch tracking
            sa.Table(
                'classifications', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('text_id', sa.String),
                sa.Column('similarity_score', sa.Float),
                sa.Column('metric', sa.String),
                sa.Column('confidence', sa.Float),
                sa.Column('label', sa.Boolean),
                sa.Column('batch_id', sa.String),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
            )
            
            metadata.create_all(self.engine)
            self.logger.info("Database tables initialized successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error initializing tables: {str(e)}")
            raise

    def store_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, bool]):
        session = self.SessionLocal()
        try:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            insert_data = [
                {
                    'text_id': id,
                    'embedding': json.dumps(emb),
                    'is_true': is_true,
                    'batch_id': batch_id
                }
                for id, (emb, is_true) in zip(metadata['ids'], zip(embedding_lists, metadata['is_true']))
            ]
            
            session.execute(
                sa.text("""
                    INSERT INTO embeddings (text_id, embedding, is_true, batch_id) 
                    VALUES (:text_id, :embedding, :is_true, :batch_id)
                    ON CONFLICT (text_id) DO UPDATE 
                    SET embedding = EXCLUDED.embedding,
                        is_true = EXCLUDED.is_true,
                        batch_id = EXCLUDED.batch_id
                """),
                insert_data
            )
            
            session.commit()
            self.logger.info(f"Stored {len(insert_data)} embeddings in batch {batch_id}")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing embeddings: {str(e)}")
            raise
        finally:
            session.close()

    def store_results(self, results: List[Dict]):
        session = self.SessionLocal()
        try:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            insert_data = [
                {
                    'text_id': r['text_id'],
                    'similarity_score': float(r['similarity_score']),
                    'metric': r['metric'],
                    'confidence': float(r['confidence']),
                    'label': bool(r['label']),
                    'batch_id': batch_id
                }
                for r in results
            ]
            
            # Now we append instead of update
            session.execute(
                sa.text("""
                    INSERT INTO classifications 
                    (text_id, similarity_score, metric, confidence, label, batch_id)
                    VALUES (:text_id, :similarity_score, :metric, :confidence, :label, :batch_id)
                """),
                insert_data
            )
            
            session.commit()
            self.logger.info(f"Stored {len(insert_data)} classification results in batch {batch_id}")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing results: {str(e)}")
            raise
        finally:
            session.close()

    def query_by_similarity(self, 
                          min_score: float = 0.0, 
                          max_score: float = 1.0,
                          metric: str = 'cosine',
                          latest_only: bool = False) -> pd.DataFrame:
        try:
            base_query = """
                SELECT 
                    text_id,
                    similarity_score,
                    metric,
                    confidence,
                    label,
                    batch_id,
                    created_at
                FROM classifications
                WHERE metric = :metric
                AND similarity_score BETWEEN :min_score AND :max_score
            """
            
            if latest_only:
                query = base_query + """
                    AND (text_id, created_at) IN (
                        SELECT text_id, MAX(created_at)
                        FROM classifications
                        WHERE metric = :metric
                        GROUP BY text_id
                    )
                """
            else:
                query = base_query
                
            query += " ORDER BY created_at DESC, similarity_score DESC"
            
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
                
                if 'similarity_score' in df.columns:
                    df[f'{metric}_score'] = df['similarity_score']
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error querying similarities: {str(e)}")
            return pd.DataFrame(columns=[
                'text_id', 
                'similarity_score', 
                f'{metric}_score',
                'metric', 
                'confidence', 
                'label',
                'batch_id',
                'created_at'
            ])

    def get_total_processed_count(self) -> int:
        """Get total number of unique documents processed"""
        try:
            query = "SELECT COUNT(DISTINCT text_id) FROM classifications"
            with self.engine.connect() as connection:
                result = connection.execute(sa.text(query)).scalar()
                return result or 0
        except Exception as e:
            self.logger.error(f"Error getting total count: {str(e)}")
            return 0

    def get_batch_stats(self) -> pd.DataFrame:
        """Get statistics by batch"""
        try:
            query = """
                SELECT 
                    batch_id,
                    COUNT(DISTINCT text_id) as doc_count,
                    MIN(created_at) as batch_start,
                    MAX(created_at) as batch_end
                FROM classifications
                GROUP BY batch_id
                ORDER BY MIN(created_at) DESC
            """
            with self.engine.connect() as connection:
                return pd.read_sql(sa.text(query), connection)
        except Exception as e:
            self.logger.error(f"Error getting batch stats: {str(e)}")
            return pd.DataFrame()

    def get_true_embeddings(self) -> np.ndarray:
        try:
            query = "SELECT embedding FROM embeddings WHERE is_true = True"
            df = pd.read_sql(query, self.engine)
            return np.array([
                np.array(json.loads(emb)) for emb in df['embedding']
            ])
        except Exception as e:
            self.logger.error(f"Error retrieving TRUE embeddings: {str(e)}")
            raise

    def close_connection(self):
        try:
            self.engine.dispose()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}")

if __name__ == "__main__":
    from config import config
    db_manager = DatabaseManager(config)
    
    # Test embedding storage
    test_embeddings = np.random.rand(5, 10)
    test_metadata = {
        'ids': [f'test_{i}' for i in range(5)],
        'is_true': [True] * 5
    }
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
    db_manager.store_results(test_results)
    
    # Test querying
    results = db_manager.query_by_similarity(metric='cosine')
    print("\nQuery Results:")
    print(results)
    
    print("\nTotal processed count:", db_manager.get_total_processed_count())
    print("\nBatch statistics:")
    print(db_manager.get_batch_stats())
    
    db_manager.close_connection()