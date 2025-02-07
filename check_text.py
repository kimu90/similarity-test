import sqlalchemy as sa
import pandas as pd
import json

def show_true_set_sample():
    """
    Show samples from the true set
    """
    try:
        print("Connecting to database...")
        db_url = "postgresql://user:password@localhost:5432/similarity_db"
        engine = sa.create_engine(db_url)
        
        query = """
            WITH TrueSetEmbs AS (
                SELECT DISTINCT ON (text_id)
                    text_id,
                    embedding,
                    created_at
                FROM embeddings
                WHERE is_true = true
                ORDER BY text_id, created_at DESC
            )
            SELECT * FROM TrueSetEmbs
            ORDER BY created_at DESC
            LIMIT 5;
        """
        
        with engine.connect() as conn:
            print("\n=== True Set Sample ===")
            df = pd.read_sql(sa.text(query), conn)
            for _, row in df.iterrows():
                print(f"\nText ID: {row['text_id']}")
                embedding = json.loads(row['embedding'])
                print(f"Embedding (first 10 values): {embedding[:10]}")
                print(f"Embedding length: {len(embedding)}")
                print(f"Created at: {row['created_at']}")
                print("-" * 80)
            
            # Also get some metrics about embeddings
            metrics_query = """
                SELECT 
                    COUNT(DISTINCT text_id) as unique_true_texts,
                    COUNT(DISTINCT latest_batch_id) as batch_count,
                    MIN(created_at) as first_created,
                    MAX(created_at) as last_created
                FROM embeddings
                WHERE is_true = true;
            """
            metrics = pd.read_sql(sa.text(metrics_query), conn)
            print("\n=== True Set Metrics ===")
            print(metrics.to_string())
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    show_true_set_sample()