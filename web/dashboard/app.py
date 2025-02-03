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
        selected_metric = st.sidebar.selectbox(
            "Similarity Metric",
            ["cosine", "jaccard", "lcs"],
            format_func=lambda x: x.capitalize()
        )
        threshold = st.sidebar.slider(f"{selected_metric.capitalize()} Similarity Threshold", 0.0, 1.0, 0.8, 0.05)
        batch_size = st.sidebar.number_input("Batch Size", 10, 1000, 100)
        
        # Load data button
        if st.sidebar.button("Load and Process New Data"):
            with st.spinner("Loading and processing data..."):
                self.process_new_data(batch_size, selected_metric)

        # Main content layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Similarity Distribution")
            self.show_similarity_distribution(threshold, selected_metric)
            
            st.subheader("Classification Metrics")
            self.show_classification_metrics(threshold, selected_metric)
            
        with col2:
            st.subheader("Top Similar Texts")
            self.show_top_similar_texts(selected_metric)
            
            st.subheader("Results Overview")
            self.show_results_table(threshold, selected_metric)

    def process_new_data(self, batch_size: int, selected_metric: str):
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
            cleaned_true = self.cleaner.batch_process(true_df['text'].tolist())
            cleaned_new = self.cleaner.batch_process(new_df['text'].tolist())
            
            # Log cleaned text sizes
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
            
            # Calculate similarities for selected metric only
            similarities = self.similarity.batch_similarities(
                new_embeddings, 
                true_embeddings,
                metric=selected_metric
            )
            
            # Log similarities
            st.write(f"{selected_metric.capitalize()} similarities shape: {similarities.shape}")
            
            # Store results for selected metric
            self.similarity.store_results(
                new_df['lens_id'].tolist(), 
                similarities,
                metric=selected_metric
            )
            
            st.success(f"Processed {len(new_df)} new texts successfully!")
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Detailed Traceback:")
            st.code(traceback.format_exc())

    def show_similarity_distribution(self, threshold: float, metric: str):
        """Display similarity score distribution"""
        try:
            results = self.db.query_by_similarity(metric=metric)
            if not results.empty:
                # Use 'similarity_score' instead of 'cosine_score'
                fig = self.visualizer.plot_distribution(
                    results['similarity_score'].values,
                    threshold,
                    f"{metric.capitalize()} Similarity Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No similarity data available. Load data first.")
        except Exception as e:
            st.error(f"Error showing distribution: {str(e)}")

    def show_classification_metrics(self, threshold: float, metric: str):
        """Display classification metrics"""
        try:
            results = self.db.query_by_similarity(metric=metric)
            if not results.empty:
                predictions = results['similarity_score'] >= threshold
                metrics = {
                    "Total Processed": len(results),
                    "Similar Texts": sum(predictions),
                    "Different Texts": len(predictions) - sum(predictions),
                    f"Avg {metric.capitalize()} Score": results['similarity_score'].mean(),
                    "Avg Confidence": results['confidence'].mean()
                }
                
                cols = st.columns(len(metrics))
                for col, (metric_name, value) in zip(cols, metrics.items()):
                    col.metric(metric_name, f"{value:.2f}")
            else:
                st.warning("No classification data available. Load data first.")
        except Exception as e:
            st.error(f"Error showing metrics: {str(e)}")

    def show_results_table(self, threshold: float, metric: str):
        """Display results table"""
        try:
            results = self.db.query_by_similarity(metric=metric)
            if not results.empty:
                # Add classification column
                results[f'is_similar'] = results['similarity_score'] >= threshold
                
                # Show filtered dataframe
                st.dataframe(
                    results[[
                        'text_id', 
                        'similarity_score',
                        'confidence',
                        'is_similar'
                    ]],
                    use_container_width=True,
                    height=400
                )
                
                # Add download button
                csv = results.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "similarity_results.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.warning("No results available. Load data first.")
        except Exception as e:
            st.error(f"Error showing results table: {str(e)}")
    def show_top_similar_texts(self, metric: str):
        """Display most similar texts"""
        try:
            results = self.db.query_by_similarity(metric=metric)
            if not results.empty:
                score_column = f"{metric}_score"
                top_similar = results.nlargest(5, score_column)
                
                for _, row in top_similar.iterrows():
                    with st.expander(f"Text ID: {row['text_id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{metric.capitalize()} Score", 
                                    f"{row[score_column]:.3f}")
                        with col2:
                            st.metric("Confidence", f"{row['confidence']:.3f}")
            else:
                st.warning("No similar texts found. Load data first.")
        except Exception as e:
            st.error(f"Error showing top similar texts: {str(e)}")

 
 

def main():
    app = DashboardApp()
    app.main()

if __name__ == "__main__":
    main()