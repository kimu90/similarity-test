# web/dashboard/app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import TextPreprocessor
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.centroid import CentroidCalculator
from src.analysis.similarity import SimilarityAnalyzer
from src.analysis.classifier import TextClassifier
from src.storage.db_manager import DatabaseManager
import pandas as pd
import numpy as np
from typing import Dict, List
from config import config

class DashboardApp:
   def __init__(self):
       """Initialize dashboard components"""
       st.set_page_config(page_title="Text Similarity Analysis", layout="wide")
       
       # Initialize components
       self.loader = DataLoader()
       self.cleaner = TextPreprocessor(config)
       self.generator = EmbeddingGenerator(config)
       self.centroid_calc = CentroidCalculator(config)
       self.similarity = SimilarityAnalyzer(config)
       self.classifier = TextClassifier(config)
       self.db = DatabaseManager(config)

   def main(self):
       """Main dashboard layout and logic"""
       st.title("Text Similarity Analysis Dashboard")

       # Sidebar controls
       st.sidebar.header("Controls")
       threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05)
       batch_size = st.sidebar.number_input("Batch Size", 10, 1000, 100)
       
       # Load data button
       if st.sidebar.button("Load New Data"):
           with st.spinner("Loading and processing data..."):
               self.process_new_data(batch_size)

       # Main content
       col1, col2 = st.columns(2)
       
       with col1:
           self.show_similarity_distribution()
           self.show_classification_metrics(threshold)
           
       with col2:
           self.show_top_similar_texts()
           self.show_results_table(threshold)

   def process_new_data(self, batch_size: int):
       """Process new batch of data"""
       try:
           # Load data
           true_df = self.loader.load_true_set()
           new_df = self.loader.load_new_texts(batch_size)
           
           # Clean texts
           cleaned_true = self.cleaner.batch_process(true_df['texts'].tolist())
           cleaned_new = self.cleaner.batch_process(new_df['texts'].tolist())
           
           # Generate embeddings
           true_embeddings = self.generator.batch_generate(cleaned_true)
           new_embeddings = self.generator.batch_generate(cleaned_new)
           
           # Calculate similarities
           similarities = self.similarity.batch_similarities(new_embeddings, true_embeddings)
           
           # Store results
           self.store_results(new_df['lens_id'].tolist(), similarities)
           
           st.success("Data processed successfully!")
           
       except Exception as e:
           st.error(f"Error processing data: {str(e)}")

   def show_similarity_distribution(self):
       """Display similarity score distribution"""
       st.subheader("Similarity Distribution")
       
       results = self.db.query_by_similarity()
       if not results.empty:
           hist_data = results['similarity_score']
           st.plotly_chart(self.create_histogram(hist_data))

   def show_classification_metrics(self, threshold: float):
       """Display classification metrics"""
       st.subheader("Classification Metrics")
       
       results = self.db.query_by_similarity()
       if not results.empty:
           predictions = (results['similarity_score'] >= threshold)
           metrics = {
               "Above Threshold": sum(predictions),
               "Below Threshold": len(predictions) - sum(predictions),
               "Average Similarity": results['similarity_score'].mean(),
               "Average Confidence": results['confidence'].mean()
           }
           
           for metric, value in metrics.items():
               st.metric(metric, f"{value:.2f}")

   def show_top_similar_texts(self):
       """Display most similar texts"""
       st.subheader("Most Similar Texts")
       
       results = self.db.query_by_similarity()
       if not results.empty:
           top_similar = results.nlargest(5, 'similarity_score')
           for _, row in top_similar.iterrows():
               st.write(f"**ID:** {row['lens_id']}")
               st.write(f"Similarity: {row['similarity_score']:.3f}")
               st.write("---")

   def show_results_table(self, threshold: float):
       """Display results table"""
       st.subheader("Results Table")
       
       results = self.db.query_by_similarity()
       if not results.empty:
           results['is_similar'] = results['similarity_score'] >= threshold
           st.dataframe(results)

   def create_histogram(self, data):
       """Create similarity distribution histogram"""
       import plotly.figure_factory as ff
       
       fig = ff.create_distplot([data], ['Similarity Scores'])
       fig.update_layout(title_text='Similarity Score Distribution')
       return fig

   def store_results(self, lens_ids: List[str], similarities: np.ndarray):
       """Store processing results"""
       max_similarities = similarities.max(axis=1)
       classifications = self.classifier.batch_classify(max_similarities, lens_ids)
       self.db.store_results([vars(c) for c in classifications])

if __name__ == "__main__":
   app = DashboardApp()
   app.main()