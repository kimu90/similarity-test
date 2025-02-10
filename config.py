import os


config = {
    'paths': {
        'true_set_path': 'dataset/true_claim.csv',
        'new_texts_path': 'dataset/false_claim.csv',

        'concat_true_set_path': 'dataset/conc_true_set.csv',  
        'concat_new_texts_path': 'dataset/conc_new_texts.csv',
        'embeddings_output_dir': 'dataset/embeddings'
    },
    'database': {
        'dbname': 'similarity_db',
        'user': 'user',
        'password': 'password',
        'host': 'db',
        'port': 5432
    },
    'data': {
        'batch_size': 4000
    },
    'model_name': 'all-MiniLM-L6-v2',
    'similarity_threshold': 0.8,
    'threshold': 0.8,
    'min_confidence': 0.6
}