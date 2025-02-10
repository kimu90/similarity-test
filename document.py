import pandas as pd
import sqlalchemy as sa
from datetime import datetime

# Database connection configuration
config = {
    'database': {
        'user': 'user',
        'password': 'password',
        'host': 'localhost',
        'port': '5432',
        'dbname': 'similarity_db'
    }
}

def export_scores():
    # Create database URL
    db_url = (
        f"postgresql://{config['database']['user']}:"
        f"{config['database']['password']}@"
        f"{config['database']['host']}:"
        f"{config['database']['port']}/"
        f"{config['database']['dbname']}"
    )
    
    # Create engine
    engine = sa.create_engine(db_url)
    
    try:
        # Query to get all classification scores
        query = """
            SELECT 
                text_id,
                similarity_score,
                confidence,
                label,
                metric,
                latest_batch_id,
                created_at
            FROM classifications
            ORDER BY created_at DESC
        """
        
        # Read data into DataFrame
        df = pd.read_sql(query, engine)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classification_scores_{timestamp}.csv"
        
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Scores exported successfully to {filename}")
        
    except Exception as e:
        print(f"Error exporting scores: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    export_scores()