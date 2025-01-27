# config.py
from pathlib import Path

# Get base project directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
TRUE_SET_DIR = DATA_DIR / 'true_set'
NEW_TEXTS_DIR = DATA_DIR / 'new_texts'

config = {
   'paths': {
       'true_set_path': TRUE_SET_DIR / 'true_set.csv',
       'new_texts_path': NEW_TEXTS_DIR / 'new_texts.csv'
   },
   'data': {
       'batch_size': 100,
       'sample_size': {
           'true_set': 100,
           'new_texts': 1000
       }
   }
}