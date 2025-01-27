# utils/generate_sample_data.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

import pandas as pd
import numpy as np
from config import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_true_set(size: int = 100) -> pd.DataFrame:
   """Generate sample TRUE set data with plant trait descriptions"""
   try:
       native_traits = [
           'drought resistance', 'pest tolerance', 'cold hardiness', 
           'soil adaptation', 'growth rate', 'root system',
           'seed dispersal', 'flowering pattern', 'nutrient efficiency'
       ]
       
       data = {
           'lens_id': [f'TRUE_{i:04d}' for i in range(size)],
           'texts': [
               f"This plant species demonstrates {np.random.choice(native_traits)} "
               f"and {np.random.choice(native_traits)} as key native traits. "
               f"Research sample {i}" 
               for i in range(size)
           ]
       }
       
       df = pd.DataFrame(data)
       logger.info(f"Generated {size} TRUE set samples")
       return df
       
   except Exception as e:
       logger.error(f"Error generating TRUE set: {str(e)}")
       raise

def generate_new_texts(size: int = 1000) -> pd.DataFrame:
   """Generate sample new texts with mix of relevant and irrelevant content"""
   try:
       # Create text templates
       similar_template = [
           "The plant exhibits {} and {} in its natural habitat.",
           "Research indicates presence of {} and {} in this species.",
           "Studies show native characteristics including {} and {}.",
       ]
       
       different_template = [
           "General article about technology and innovation.",
           "News report on economic developments.",
           "Scientific study on climate patterns.",
           "Research paper on animal behavior.",
       ]
       
       native_traits = [
           'drought resistance', 'pest tolerance', 'cold hardiness', 
           'soil adaptation', 'growth rate', 'root system'
       ]
       
       # Generate similar texts (30% of total)
       similar_size = size // 3
       similar_texts = [
           np.random.choice(similar_template).format(
               np.random.choice(native_traits),
               np.random.choice(native_traits)
           )
           for _ in range(similar_size)
       ]
       
       # Generate different texts
       different_texts = [
           f"{np.random.choice(different_template)} Sample {i}"
           for i in range(size - similar_size)
       ]
       
       data = {
           'lens_id': [f'NEW_{i:04d}' for i in range(size)],
           'texts': similar_texts + different_texts
       }
       
       df = pd.DataFrame(data)
       # Shuffle the data
       df = df.sample(frac=1).reset_index(drop=True)
       
       logger.info(f"Generated {size} new text samples ({similar_size} similar, {size-similar_size} different)")
       return df
       
   except Exception as e:
       logger.error(f"Error generating new texts: {str(e)}")
       raise

def generate_sample_data():
   """Generate and save both datasets"""
   try:
       # Create directories if they don't exist
       config['paths']['true_set_path'].parent.mkdir(parents=True, exist_ok=True)
       config['paths']['new_texts_path'].parent.mkdir(parents=True, exist_ok=True)
       
       logger.info("Starting sample data generation...")
       
       # Generate data
       true_df = generate_true_set(config['data']['sample_size']['true_set'])
       new_df = generate_new_texts(config['data']['sample_size']['new_texts'])
       
       # Save to configured paths
       true_df.to_csv(config['paths']['true_set_path'], index=False)
       new_df.to_csv(config['paths']['new_texts_path'], index=False)
       
       logger.info(f"Successfully generated and saved:")
       logger.info(f"- TRUE set: {len(true_df)} samples at {config['paths']['true_set_path']}")
       logger.info(f"- New texts: {len(new_df)} samples at {config['paths']['new_texts_path']}")
       
       # Print some sample statistics
       logger.info("\nData Statistics:")
       logger.info(f"TRUE set unique texts: {true_df['texts'].nunique()}")
       logger.info(f"New texts unique content: {new_df['texts'].nunique()}")
       
   except Exception as e:
       logger.error(f"Error in sample data generation: {str(e)}")
       raise

if __name__ == "__main__":
   try:
       generate_sample_data()
       logger.info("Sample data generation completed successfully")
   except Exception as e:
       logger.error(f"Failed to generate sample data: {str(e)}")
       sys.exit(1)