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
    def __init__(self, config=None):
        try:
            if config is None:
                from config import config

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
            
            # Embeddings table with composite unique constraint
            sa.Table(
                'embeddings', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('text_id', sa.String),
                sa.Column('embedding', sa.JSON),
                sa.Column('is_true', sa.Boolean),
                sa.Column('latest_batch_id', sa.String),
                sa.Column('metric', sa.String),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
                sa.UniqueConstraint('text_id', 'metric', name='uix_text_metric')
            )
            
            # Classifications table with composite unique constraint
            sa.Table(
                'classifications', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('text_id', sa.String),
                sa.Column('similarity_score', sa.Float),
                sa.Column('metric', sa.String),
                sa.Column('confidence', sa.Float),
                sa.Column('label', sa.Boolean),
                sa.Column('latest_batch_id', sa.String),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
                sa.UniqueConstraint('text_id', 'metric', 'created_at', name='uix_classification')
            )
            
            # Processing tracker table
            sa.Table(
                'processing_tracker', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('last_processed_total_processed', sa.Integer, nullable=False, default=0),  # Corrected column definition
                sa.Column('total_rows', sa.Integer, default=0),
                sa.Column('latest_batch_id', sa.String),
                sa.Column('status', sa.String),
                sa.Column('metric', sa.String),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
                sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
            )
            
            # Create indexes for performance
            sa.Index('idx_classifications_metric', 'metric')
            sa.Index('idx_classifications_text_id', 'text_id')
            
            metadata.create_all(self.engine)
            self.logger.info("Database tables initialized successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error initializing tables: {str(e)}")
            raise
    def get_processing_status(self, metric: Optional[str] = None) -> Dict:
        try:
            query = """
                SELECT 
                    MAX(last_processed_total_processed) AS total_processed,
                    MAX(total_rows) AS total_rows,
                    MAX(status) AS status,
                    MAX(latest_batch_id) AS latest_latest_batch_id,
                    metric
                FROM processing_tracker
                WHERE metric = :metric
                GROUP BY metric
                ORDER BY MAX(updated_at) DESC
                LIMIT 1
            """
            with self.engine.connect() as connection:
                result = connection.execute(sa.text(query), {'metric': metric}).fetchone()
                if result:
                    return {
                        'total_processed': result[0] or 0,
                        'total_rows': result[1] or 0,
                        'status': result[2] or 'not_started',
                        'latest_latest_batch_id': result[3],
                        'metric': result[4]
                    }
                return {
                    'total_processed': 0,
                    'total_rows': 0,
                    'status': 'not_started',
                    'latest_latest_batch_id': None,
                    'metric': metric
                }
        except Exception as e:
            self.logger.error(f"Error getting processing status: {str(e)}")
            return {
                'total_processed': 0,
                'total_rows': 0,
                'status': 'failed', 
                'latest_latest_batch_id': None,
                'metric': metric
            }
    def update_processing_status(self, total_processed: int, total_rows: int, status: str, metric: str):
        session = self.SessionLocal()
        try:
            latest_batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            query = """
                INSERT INTO processing_tracker 
                (last_processed_total_processed, total_rows, status, latest_batch_id, metric)
                VALUES (:total_processed, :total_rows, :status, :latest_batch_id, :metric)
            """
            
            session.execute(
                sa.text(query),
                {
                    'total_processed': max(total_processed, 0),
                    'total_rows': max(total_rows, 0),
                    'status': status or 'in_progress',
                    'latest_batch_id': latest_batch_id,
                    'metric': metric
                }
            )
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error updating processing status: {str(e)}")
            raise
        finally:
            session.close()
    def get_batch_stats(self) -> pd.DataFrame:
        try:
            query = """
                SELECT 
                    latest_batch_id,
                    COUNT(DISTINCT text_id) as doc_count,
                    MIN(created_at) as batch_start,
                    MAX(created_at) as batch_end,
                    (SELECT DISTINCT metric FROM classifications 
                     WHERE latest_batch_id = combined_batches.latest_batch_id LIMIT 1) as metric
                FROM (
                    SELECT text_id, latest_batch_id, created_at, 'classifications' as source
                    FROM classifications
                    UNION
                    SELECT text_id, latest_batch_id, created_at, 'embeddings' as source
                    FROM embeddings
                ) combined_batches
                GROUP BY latest_batch_id
                ORDER BY batch_start DESC
            """
            with self.engine.connect() as connection:
                return pd.read_sql(sa.text(query), connection)
        except Exception as e:
            self.logger.error(f"Error getting batch stats: {str(e)}")
            return pd.DataFrame(columns=[
                'latest_batch_id', 'doc_count', 'batch_start', 'batch_end', 'metric'
            ])

    def store_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, bool], metric: Optional[str] = None):
        session = self.SessionLocal()
        try:
            latest_batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            insert_data = [
                {
                    'text_id': id,
                    'embedding': json.dumps(emb),
                    'is_true': is_true,
                    'latest_batch_id': latest_batch_id,
                    'metric': metric
                }
                for id, (emb, is_true) in zip(metadata['ids'], zip(embedding_lists, metadata['is_true']))
            ]
            
            session.execute(
                sa.text("""
                    INSERT INTO embeddings 
                    (text_id, embedding, is_true, latest_batch_id, metric) 
                    VALUES (:text_id, :embedding, :is_true, :latest_batch_id, :metric)
                    ON CONFLICT (text_id, metric) DO UPDATE 
                    SET embedding = EXCLUDED.embedding,
                        is_true = EXCLUDED.is_true,
                        latest_batch_id = EXCLUDED.latest_batch_id
                """),
                insert_data
            )
            
            session.commit()
            self.logger.info(f"Stored {len(insert_data)} embeddings in batch {latest_batch_id}")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing embeddings: {str(e)}")
            raise
        finally:
            session.close()

    def store_results(self, results: List[Dict]):
        session = self.SessionLocal()
        try:
            latest_batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            insert_data = [
                {
                    'text_id': r['text_id'],
                    'similarity_score': float(r['similarity_score']),
                    'metric': r['metric'],
                    'confidence': float(r['confidence']),
                    'label': bool(r['label']),
                    'latest_batch_id': latest_batch_id
                }
                for r in results
            ]
            
            session.execute(
                sa.text("""
                    INSERT INTO classifications 
                    (text_id, similarity_score, metric, confidence, label, latest_batch_id)
                    VALUES (:text_id, :similarity_score, :metric, :confidence, :label, :latest_batch_id)
                    ON CONFLICT (text_id, metric, created_at) DO NOTHING
                """),
                insert_data
            )
            
            session.commit()
            self.logger.info(f"Stored {len(insert_data)} classification results in batch {latest_batch_id}")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing results: {str(e)}")
            raise
        finally:
            session.close()

    def query_by_similarity(self, min_score: float = 0.0, max_score: float = 1.0,
                         min_confidence: float = 0.0, max_confidence: float = 1.0,
                         metric: str = 'cosine', latest_only: bool = False) -> pd.DataFrame:
        try:
            base_query = """
                SELECT 
                    text_id, similarity_score, metric, confidence, label, latest_batch_id, created_at
                FROM classifications
                WHERE metric = :metric
                AND similarity_score BETWEEN :min_score AND :max_score
                AND confidence BETWEEN :min_confidence AND :max_confidence
            """
            
            if latest_only:
                query = base_query + """
                    AND (text_id, metric, created_at) IN (
                        SELECT text_id, metric, MAX(created_at)
                        FROM classifications
                        WHERE metric = :metric
                        GROUP BY text_id, metric
                    )
                """
            else:
                query = base_query
                
            query += " ORDER BY created_at DESC, similarity_score DESC"
            
            with self.engine.connect() as connection:
                return pd.read_sql(
                    sa.text(query), 
                    connection, 
                    params={
                        'metric': metric, 
                        'min_score': min_score, 
                        'max_score': max_score,
                        'min_confidence': min_confidence,
                        'max_confidence': max_confidence
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error querying similarities: {str(e)}")
            return pd.DataFrame(columns=[
                'text_id', 'similarity_score', 'metric',
                'confidence', 'label', 'latest_batch_id', 'created_at'
            ])

    def get_total_processed_count(self) -> int:
        try:
            query = "SELECT COUNT(DISTINCT text_id) FROM classifications"
            with self.engine.connect() as connection:
                result = connection.execute(sa.text(query)).scalar()
                return result or 0
        except Exception as e:
            self.logger.error(f"Error getting total count: {str(e)}")
            return 0

    def get_true_embeddings(self) -> np.ndarray:
        try:
            query = "SELECT embedding FROM embeddings WHERE is_true = True"
            df = pd.read_sql(query, self.engine)
            return np.array([np.array(json.loads(emb)) for emb in df['embedding']])
        except Exception as e:
            self.logger.error(f"Error retrieving TRUE embeddings: {str(e)}")
            raise

    def close_connection(self):
        try:
            self.engine.dispose()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}")