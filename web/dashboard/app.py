# web/dashboard/app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from typing import Dict
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import logging
import traceback
import io
import time
from datetime import datetime

from src.preprocessing.loader import DataLoader
from src.preprocessing.cleaner import TextPreprocessor
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.centroid import CentroidCalculator
from src.analysis.similarity import SimilarityAnalyzer
from src.analysis.classifier import TextClassifier
from src.storage.db_manager import DatabaseManager
from web.dashboard.metrics import MetricsCalculator
from web.dashboard.plots import Visualizer
import xlsxwriter

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
        self.start_time = time.time()

    def main(self):
        """Main dashboard layout and logic"""
        st.title("Text Similarity Analysis Dashboard")

        # Create tabs for main dashboard and report
        main_tab, report_tab = st.tabs(["Analysis Dashboard", "Comprehensive Report"])

        with main_tab:
            # Sidebar controls
            st.sidebar.header("Controls")
            selected_metric = st.sidebar.selectbox(
                "Similarity Metric",
                ["cosine", "jaccard", "lcs"],
                format_func=lambda x: x.capitalize()
            )
            threshold = st.sidebar.slider(f"{selected_metric.capitalize()} Similarity Threshold", 0.0, 1.0, 0.8, 0.05)
            batch_size = st.sidebar.number_input("Batch Size", 100, 8000, 2500)
            
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

        with report_tab:
            self.show_comprehensive_report()

    def compute_correlation_matrix(self):
        """
        Compute a correlation matrix between true set and new texts
        
        Returns:
            np.ndarray: Correlation matrix of similarities
        """
        try:
            # Combine texts from both datasets
            all_texts = pd.concat([
                self.true_set_df['text'], 
                self.new_texts_df['text']
            ])

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            
            # Compute TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Compute cosine similarity matrix
            correlation_matrix = cosine_similarity(tfidf_matrix)
            
            # Store the matrix as an instance attribute
            self.correlation_matrix = correlation_matrix
            
            return correlation_matrix
        
        except Exception as e:
            print(f"Error computing correlation matrix: {e}")
            return None


    def show_similarity_distribution(self, threshold: float, metric: str):
        """Display similarity score distribution for specific metric"""
        try:
            # Explicitly query results for the selected metric
            results = self.db.query_by_similarity(metric=metric)
            
            if not results.empty:
                fig = self.visualizer.plot_distribution(
                    results['similarity_score'].values,
                    threshold,
                    f"{metric.capitalize()} Similarity Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No {metric} similarity data available. Load data first.")
        except Exception as e:
            st.error(f"Error showing {metric} distribution: {str(e)}")


    def show_top_similar_texts(self, metric: str):
        """Display most similar texts for specific metric"""
        try:
            # Query results specifically for the selected metric
            results = self.db.query_by_similarity(metric=metric)
            
            if not results.empty:
                top_similar = results.nlargest(5, 'similarity_score')
                for _, row in top_similar.iterrows():
                    with st.expander(f"Text ID: {row['text_id']} ({metric.capitalize()} Similarity)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{metric.capitalize()} Score", 
                                    f"{row['similarity_score']:.3f}")
                        with col2:
                            st.metric("Confidence", f"{row['confidence']:.3f}")
            else:
                st.warning(f"No similar texts found for {metric} metric. Load data first.")
        except Exception as e:
            st.error(f"Error showing top similar {metric} texts: {str(e)}")
    def show_comprehensive_report(self):
        # Get metrics from recent batch processing
        batch_stats = self.db.get_batch_stats()
        
        # Collect unique metrics from batch stats
        processed_metrics = batch_stats['metric'].dropna().unique().tolist() if not batch_stats.empty else ['cosine']
        
        # If no metrics found, default to cosine
        if not processed_metrics:
            processed_metrics = ['cosine']
        
        st.header("Comprehensive Similarity Analysis Reports")
        
        for metric in processed_metrics:
            # Create a separate tab or section for each metric
            st.subheader(f"{metric.capitalize()} Similarity Report")
            
            # Get results for this specific metric
            results = self.db.query_by_similarity(metric=metric)
            
            if results.empty:
                st.warning(f"No {metric} similarity results found.")
                continue
            
            # Processing Status Section
            status = self.db.get_processing_status()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", status['total_rows'])
                st.metric("Processed Documents", status['offset'])
            with col2:
                if status['total_rows'] > 0:
                    progress = (status['offset'] / status['total_rows']) * 100
                    st.metric("Progress", f"{progress:.1f}%")
                st.metric("Status", status['status'].title())
            with col3:
                st.metric("Remaining Documents", status['total_rows'] - status['offset'])
                if status['batch_id']:
                    st.metric("Current Batch", status['batch_id'])
            
            # Similarity Score Analysis
            data = results['similarity_score']
            
            # Distribution Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Violin plot
                fig_violin = go.Figure(data=go.Violin(
                    y=data,
                    name=metric.capitalize(),
                    box_visible=True,
                    meanline_visible=True
                ))
                fig_violin.update_layout(
                    title=f"{metric.capitalize()} Similarity Distribution",
                    yaxis_title="Similarity Score"
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            
            with col2:
                # Threshold analysis
                thresholds = np.linspace(0, 1, 20)
                ratios = [(data >= threshold).mean() for threshold in thresholds]
                
                fig_threshold = go.Figure(data=go.Scatter(
                    x=thresholds,
                    y=ratios,
                    mode='lines+markers',
                    name=metric.capitalize()
                ))
                fig_threshold.update_layout(
                    title="Document Ratio vs Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="Ratio of Documents"
                )
                st.plotly_chart(fig_threshold, use_container_width=True)
            
            # Detailed Statistics
            st.subheader("Detailed Statistics")
            stats = {
                'Mean': data.mean(),
                'Median': data.median(),
                'Std Dev': data.std(),
                'Skewness': data.skew(),
                'Kurtosis': data.kurtosis(),
                'Min': data.min(),
                'Max': data.max(),
                '25th Percentile': data.quantile(0.25),
                '75th Percentile': data.quantile(0.75),
                'Above 0.8': (data >= 0.8).mean(),
                'Below 0.2': (data < 0.2).mean()
            }
            
            st.table(pd.DataFrame.from_dict(stats, orient='index', columns=[metric.capitalize()]))
            
            # Export Options
            st.subheader("Export Options")
            
            # Excel Export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                results.to_excel(writer, sheet_name=f'{metric}_details', index=False)
                pd.DataFrame.from_dict(stats, orient='index', columns=[metric.capitalize()]).to_excel(writer, sheet_name='Summary_Stats')
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label=f"Download {metric.capitalize()} Report (Excel)",
                    data=buffer.getvalue(),
                    file_name=f"{metric}_similarity_report.xlsx",
                    mime="application/vnd.ms-excel"
                )
            
            # Divider between metrics
            st.markdown("---")
    def generate_summary_text(self, detailed_stats: Dict, correlation_matrix: pd.DataFrame) -> str:
        """Generate summary text report"""
        summary = ["SIMILARITY ANALYSIS SUMMARY REPORT", "=" * 30, ""]
        
        # Add timestamp
        summary.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add statistics for the processed metric
        summary.append("DETAILED STATISTICS")
        summary.append("-" * 20)
        for metric, stats in detailed_stats.items():
            summary.append(f"\n{metric.upper()} SIMILARITY:")
            for stat_name, value in stats.items():
                summary.append(f"{stat_name}: {value:.4f}")
        
        # Key findings
        summary.append("\nKEY FINDINGS")
        summary.append("-" * 20)
        for metric, stats in detailed_stats.items():
            summary.append(f"\n{metric.capitalize()} Similarity:")
            summary.append(f"- Average similarity: {stats['Mean']:.4f}")
            summary.append(f"- {stats['Above 0.8']*100:.1f}% of documents above 0.8 threshold")
            summary.append(f"- {stats['Below 0.2']*100:.1f}% of documents below 0.2 threshold")
            
            # Distribution characterization
            if stats['Skewness'] > 0:
                skew_desc = "positively skewed (more low values)"
            else:
                skew_desc = "negatively skewed (more high values)"
            summary.append(f"- Distribution is {skew_desc}")
            
            # Spread characterization
            spread = stats['75th Percentile'] - stats['25th Percentile']
            summary.append(f"- Interquartile range: {spread:.4f}")

        # Add processing information
        summary.append("\nPROCESSING INFORMATION")
        summary.append("-" * 20)
        true_embeddings, true_text_ids = self.similarity.get_true_embeddings_with_ids()
        new_embeddings, new_text_ids = self.similarity.get_new_embeddings_with_ids()
        summary.append(f"Reference Documents: {len(true_text_ids)}")
        summary.append(f"New Documents: {len(new_text_ids)}")
        summary.append(f"Total Comparisons: {len(true_text_ids) * len(new_text_ids)}")
        
        return "\n".join(summary)

        
    def show_classification_metrics(self, threshold: float, metric: str):
        try:
            # Get processing status
            status = self.db.get_processing_status()
            total_all_time = self.db.get_total_processed_count()
            
            # Get batch stats with metric filter
            batch_stats = self.db.get_batch_stats()
            filtered_batch_stats = batch_stats[batch_stats['metric'] == metric] if not batch_stats.empty else pd.DataFrame()
            
            # Get current results for specific metric
            current_results = self.db.query_by_similarity(metric=metric, latest_only=True)

            # Processing Progress Section
            st.subheader("Processing Progress")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents Processed", total_all_time)
                st.metric("Documents Remaining", status['total_rows'] - status['offset'])
            with col2:
                if status['total_rows'] > 0:
                    progress = (status['offset'] / status['total_rows']) * 100
                    st.metric("Progress", f"{progress:.1f}%")
                st.metric("Processing Status", status['status'].title())

            # Batch Statistics
            st.subheader("Batch Statistics")
            if not filtered_batch_stats.empty:
                st.metric("Total Batches", len(filtered_batch_stats))
                st.metric("Last Batch Size", filtered_batch_stats.iloc[0]['doc_count'])
                st.metric("Last Batch Date", filtered_batch_stats.iloc[0]['batch_end'].strftime('%Y-%m-%d %H:%M'))

            # Similarity Metrics
            if not current_results.empty:
                st.subheader("Similarity Analysis")
                predictions = current_results['similarity_score'] >= threshold
                metrics = {
                    "Similar Texts": sum(predictions),
                    "Different Texts": len(predictions) - sum(predictions),
                    f"Avg {metric.capitalize()} Score": current_results['similarity_score'].mean(),
                    "Avg Confidence": current_results['confidence'].mean()
                }
                
                cols = st.columns(len(metrics))
                for col, (metric_name, value) in zip(cols, metrics.items()):
                    col.metric(metric_name, f"{value:.2f}")
        except Exception as e:
            st.error(f"Error showing metrics: {str(e)}")

    def process_new_data(self, batch_size: int, selected_metric: str):
        try:
            true_df = self.loader.load_true_set()
            
            status = self.db.get_processing_status(metric=selected_metric)
            
            if status['total_rows'] == 0:
                new_df = self.loader.load_new_texts(batch_size, resume=False, selected_metric=selected_metric)
                total_rows = len(new_df)
                
                self.db.update_processing_status(
                    0, 
                    total_rows, 
                    'in_progress', 
                    metric=selected_metric
                )
                
                status = self.db.get_processing_status(metric=selected_metric)
            
            progress_bar = st.progress(0)
            st.write(f"Starting from position: {status['offset']}/{status['total_rows']}")
            
            cleaned_true = self.cleaner.batch_process(true_df['text'].tolist())
            true_embeddings = self.generator.batch_generate(
                cleaned_true,
                true_df['lens_id'].tolist(),
                is_true_set=True
            )
            
            try:
                new_df = self.loader.load_new_texts(batch_size, resume=True, selected_metric=selected_metric)
                if not new_df.empty:
                    cleaned_new = self.cleaner.batch_process(new_df['text'].tolist())
                    new_embeddings = self.generator.batch_generate(
                        cleaned_new,
                        new_df['lens_id'].tolist(),
                        is_true_set=False
                    )
                    
                    similarities = self.similarity.batch_similarities(
                        new_embeddings,
                        true_embeddings,
                        metric=selected_metric
                    )
                    
                    self.similarity.store_results(
                        new_df['lens_id'].tolist(),
                        similarities,
                        metric=selected_metric
                    )
                    
                    new_offset = status['offset'] + len(new_df)
                    progress = min(new_offset / max(status['total_rows'], 1), 1.0)
                    progress_bar.progress(progress)
                    
                    st.success(f"Processed {len(new_df)} documents ({new_offset}/{status['total_rows']} total)")
                    
                    status_label = 'completed' if new_offset >= status['total_rows'] else 'in_progress'
                    self.db.update_processing_status(
                        new_offset,
                        status['total_rows'],
                        status_label,
                        metric=selected_metric
                    )
                else:
                    st.info("All documents processed!")
                    self.db.update_processing_status(
                        status['total_rows'],
                        status['total_rows'],
                        'completed',
                        metric=selected_metric
                    )
                    
            except Exception as e:
                self.db.update_processing_status(
                    status['offset'],
                    status['total_rows'],
                    'failed',
                    metric=selected_metric
                )
                raise e
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Detailed Traceback:")
            st.code(traceback.format_exc())


    def show_results_table(self, threshold: float, metric: str):
        """Display results table for specific metric"""
        try:
            # Get latest results specifically for the selected metric
            results = self.db.query_by_similarity(
                metric=metric, 
                latest_only=True
            )
            
            if not results.empty:
                results['is_similar'] = results['similarity_score'] >= threshold
                
                st.dataframe(
                    results[[
                        'text_id',
                        'similarity_score',
                        'confidence',
                        'is_similar',
                        'batch_id',
                        'created_at'
                    ]].sort_values('created_at', ascending=False),
                    use_container_width=True,
                    height=400
                )
                
                csv = results.to_csv(index=False)
                st.download_button(
                    f"Download {metric.capitalize()} Results CSV",
                    csv,
                    f"{metric}_similarity_results.csv",
                    "text/csv",
                    key=f'{metric}-download-csv'
                )
            else:
                st.warning(f"No {metric} similarity results available. Load data first.")
        except Exception as e:
            st.error(f"Error showing {metric} results table: {str(e)}")


    


def main():
   app = DashboardApp()
   app.main()

if __name__ == "__main__":
   main()