import os

config = {
    'paths': {
        'true_set_path': 'data/true_set/true_set.csv',
        'new_texts_path': 'data/new_texts/new_texts.csv',
        'embeddings_output_dir': 'data/embeddings'
    },
    'database': {
        'dbname': 'similarity_db',
        'user': 'user',
        'password': 'password',
        'host': 'db',
        'port': 5432
    },
    'data': {
        'batch_size': 100
    },
    'model_name': 'all-MiniLM-L6-v2',
    'similarity_threshold': 0.8,
    'threshold': 0.8,
    'min_confidence': 0.6
}