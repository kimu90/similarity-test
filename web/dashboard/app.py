# web/dashboard/app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import logging
import traceback

from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import TextPreprocessor
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.centroid import CentroidCalculator
from src.analysis.similarity import SimilarityAnalyzer
from src.analysis.classifier import TextClassifier
from src.storage.db_manager import DatabaseManager
from web.dashboard.metrics import MetricsCalculator
from web.dashboard.plots import Visualizer

from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardApp:
    def __init__(self):
        """Initialize dashboard components"""
        st.set_page_config(page_title="Text Similarity Analysis", layout="wide")
        
        # Initialize database manager
        self.db = DatabaseManager(config)
        
        # Initialize components
        self.loader = DataLoader()
        self.cleaner = TextPreprocessor(config)
        self.generator = EmbeddingGenerator(config, self.db)
        self.centroid_calc = CentroidCalculator(config, self.db)
        self.similarity = SimilarityAnalyzer(config, self.db)
        self.classifier = TextClassifier(config, self.db)
        self.metrics = MetricsCalculator()
        self.visualizer = Visualizer()

    def main(self):
        """Main dashboard layout and logic"""
        st.title("Text Similarity Analysis Dashboard")

        # Sidebar controls
        st.sidebar.header("Controls")
        threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05)
        batch_size = st.sidebar.number_input("Batch Size", 10, 1000, 100)
        
        # Load data button
        if st.sidebar.button("Load and Process New Data"):
            with st.spinner("Loading and processing data..."):
                self.process_new_data(batch_size)

        # Main content layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Similarity Distribution")
            self.show_similarity_distribution(threshold)
            
            st.subheader("Classification Metrics")
            self.show_classification_metrics(threshold)
            
        with col2:
            st.subheader("Top Similar Texts")
            self.show_top_similar_texts()
            
            st.subheader("Results Overview")
            self.show_results_table(threshold)

    def process_new_data(self, batch_size: int):
        """Process new batch of data"""
        try:
            # Load data
            true_df = self.loader.load_true_set()
            new_df = self.loader.load_new_texts(batch_size)
            
            # Log original data
            st.write(f"TRUE set size: {len(true_df)}")
            st.write(f"New texts size: {len(new_df)}")
            
            # Print first few rows of new_df to inspect
            st.write("First few rows of new texts:")
            st.dataframe(new_df.head())
            
            # Clean texts
            cleaned_true = self.cleaner.batch_process(true_df['texts'].tolist())
            cleaned_new = self.cleaner.batch_process(new_df['texts'].tolist())
            
            # Log cleaned text sizes and contents
            st.write(f"Cleaned TRUE set size: {len(cleaned_true)}")
            st.write(f"Cleaned new texts size: {len(cleaned_new)}")
            
            if not cleaned_new:
                st.error("No new texts remained after cleaning. Check cleaning criteria.")
                return
            
            # Generate and store embeddings
            true_embeddings = self.generator.batch_generate(
                cleaned_true, 
                true_df['lens_id'].tolist(),
                is_true_set=True
            )
            new_embeddings = self.generator.batch_generate(
                cleaned_new, 
                new_df['lens_id'].tolist(),
                is_true_set=False
            )
            
            # Log embedding sizes
            st.write(f"TRUE embeddings shape: {true_embeddings.shape}")
            st.write(f"New embeddings shape: {new_embeddings.shape}")
            
            # Ensure 2D arrays for similarity calculation
            if new_embeddings.ndim == 1:
                new_embeddings = new_embeddings.reshape(-1, 1)
            
            # Calculate similarities
            similarities = self.similarity.batch_similarities(new_embeddings, true_embeddings)
            
            # Log similarities
            st.write(f"Similarities shape: {similarities.shape}")
            
            # Store results
            self.store_results(new_df['lens_id'].tolist(), similarities)
            
            st.success(f"Processed {len(new_df)} new texts successfully!")
            
        except Exception as e:
            import traceback
            st.error(f"Error processing data: {str(e)}")
            st.error("Detailed Traceback:")
            st.code(traceback.format_exc())

    def store_results(self, new_text_ids: list, similarities: np.ndarray):
        """Store processing results in database"""
        try:
            max_similarities = similarities.max(axis=1)
            classifications = self.classifier.batch_classify(max_similarities, new_text_ids)
            
            # Convert to dictionary for database storage
            results = [
                {
                    'text_id': result.text_id,  # Changed from lens_id to text_id
                    'similarity': result.similarity_score,
                    'confidence': result.confidence,
                    'label': result.label
                }
                for result in classifications
            ]
            
            # Store in database
            self.db.store_results(results)
            logger.info(f"Stored {len(results)} classification results")
        
        except Exception as e:
            logger.error(f"Error storing results: {str(e)}")
            st.error(f"Error storing results: {str(e)}")

    def show_similarity_distribution(self, threshold: float):
        """Display similarity score distribution"""
        try:
            results = self.db.query_by_similarity()
            if not results.empty:
                fig = self.visualizer.plot_distribution(results['similarity_score'].values)
                st.plotly_chart(fig)
            else:
                st.warning("No similarity data available. Load data first.")
        except Exception as e:
            st.error(f"Error showing distribution: {str(e)}")

    def show_classification_metrics(self, threshold: float):
        """Display classification metrics"""
        try:
            results = self.db.query_by_similarity()
            if not results.empty:
                predictions = results['similarity_score'] >= threshold
                metrics = {
                    "Total Processed": len(results),
                    "Similar Texts": sum(predictions),
                    "Different Texts": len(predictions) - sum(predictions),
                    "Avg Similarity": results['similarity_score'].mean(),
                    "Avg Confidence": results['confidence'].mean()
                }
                
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.2f}")
            else:
                st.warning("No classification data available. Load data first.")
        except Exception as e:
            st.error(f"Error showing metrics: {str(e)}")

    def show_top_similar_texts(self):
        """Display most similar texts"""
        try:
            results = self.db.query_by_similarity()
            if not results.empty:
                top_similar = results.nlargest(5, 'similarity_score')
                for _, row in top_similar.iterrows():
                    st.write(f"**Text ID:** {row['text_id']}")
                    st.write(f"Similarity: {row['similarity_score']:.3f}")
                    st.write(f"Confidence: {row['confidence']:.3f}")
                    st.write("---")
            else:
                st.warning("No similar texts found. Load data first.")
        except Exception as e:
            st.error(f"Error showing top similar texts: {str(e)}")

    def show_results_table(self, threshold: float):
        """Display results table"""
        try:
            results = self.db.query_by_similarity()
            if not results.empty:
                results['is_similar'] = results['similarity_score'] >= threshold
                st.dataframe(results)
            else:
                st.warning("No results available. Load data first.")
        except Exception as e:
            st.error(f"Error showing results table: {str(e)}")

def main():
    app = DashboardApp()
    app.main()

if __name__ == "__main__":
    main()