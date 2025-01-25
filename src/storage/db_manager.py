# src/storage/db_manager.py

import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
import logging

class DatabaseManager:
   def __init__(self, config: Dict):
       self.conn = psycopg2.connect(**config['database'])
       self.logger = logging.getLogger(__name__)
       self._init_tables()

   def _init_tables(self):
       """Initialize database tables"""
       with self.conn.cursor() as cur:
           cur.execute("""
               CREATE TABLE IF NOT EXISTS embeddings (
                   id SERIAL PRIMARY KEY,
                   text_id TEXT,
                   embedding vector(384),
                   is_true BOOLEAN
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
       self.conn.commit()

   def store_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, bool]):
       """Store embeddings with metadata"""
       with self.conn.cursor() as cur:
           execute_values(cur, 
               "INSERT INTO embeddings (text_id, embedding, is_true) VALUES %s",
               [(id, emb, is_true) for id, (emb, is_true) 
                in zip(metadata['ids'], zip(embeddings, metadata['is_true']))]
           )
       self.conn.commit()

   def store_results(self, results: List[Dict]):
       """Store classification results"""
       with self.conn.cursor() as cur:
           execute_values(cur,
               """INSERT INTO classifications 
                  (text_id, similarity_score, confidence, label)
                  VALUES %s""",
               [(r['text_id'], r['similarity'], r['confidence'], r['label']) 
                for r in results]
           )
       self.conn.commit()

   def query_by_similarity(self, min_score: float = 0.0, 
                         max_score: float = 1.0) -> pd.DataFrame:
       """Query results by similarity range"""
       query = """
           SELECT * FROM classifications 
           WHERE similarity_score BETWEEN %s AND %s
           ORDER BY similarity_score DESC
       """
       return pd.read_sql(query, self.conn, params=[min_score, max_score])

   def get_true_embeddings(self) -> np.ndarray:
       """Get all TRUE set embeddings"""
       query = "SELECT embedding FROM embeddings WHERE is_true = True"
       df = pd.read_sql(query, self.conn)
       return np.stack(df['embedding'].values)