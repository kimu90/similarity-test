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
        """Display similarity score distribution"""
        try:
            results = self.db.query_by_similarity(metric=metric)
            if not results.empty:
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


    def show_top_similar_texts(self, metric: str):
        """Display most similar texts"""
        try:
            results = self.db.query_by_similarity(metric=metric)
            if not results.empty:
                top_similar = results.nlargest(5, 'similarity_score')
                for _, row in top_similar.iterrows():
                    with st.expander(f"Text ID: {row['text_id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{metric.capitalize()} Score", 
                                    f"{row['similarity_score']:.3f}")
                        with col2:
                            st.metric("Confidence", f"{row['confidence']:.3f}")
            else:
                st.warning("No similar texts found. Load data first.")
        except Exception as e:
            st.error(f"Error showing top similar texts: {str(e)}")

    def show_comprehensive_report(self):
        correlation_matrix = self.compute_correlation_matrix()
        st.header("Comprehensive Similarity Analysis Report")
        
        # Processing Status Section
        status = self.db.get_processing_status()
        batch_stats = self.db.get_batch_stats()
        
        st.subheader("Processing Status")
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

        # Batch History
        st.subheader("Batch Processing History")
        if not batch_stats.empty:
            batch_stats['duration'] = (batch_stats['batch_end'] - batch_stats['batch_start']).dt.total_seconds()
            st.dataframe(
                batch_stats[[
                    'batch_id',
                    'doc_count',
                    'batch_start',
                    'batch_end',
                    'duration'
                ]].sort_values('batch_start', ascending=False),
                use_container_width=True
            )
            
            # Batch processing chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=batch_stats['batch_start'],
                y=batch_stats['doc_count'],
                name='Documents per Batch'
            ))
            fig.update_layout(
                title="Documents Processed per Batch",
                xaxis_title="Batch Date",
                yaxis_title="Number of Documents"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Get results for all metrics
        metrics = ['cosine', 'jaccard', 'lcs']
        all_results = {}
        for metric in metrics:
            all_results[metric] = self.db.query_by_similarity(metric=metric)

        # Similarity Score Analysis
        st.subheader("2. Similarity Score Analysis")
        
        # Distribution Analysis
        st.write("#### Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Violin plots
            fig = go.Figure()
            for metric in metrics:
                if not all_results[metric].empty:
                    fig.add_trace(go.Violin(
                        y=all_results[metric]['similarity_score'],
                        name=metric.capitalize(),
                        box_visible=True,
                        meanline_visible=True
                    ))
            fig.update_layout(
                title="Similarity Score Distributions",
                yaxis_title="Similarity Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Threshold analysis
            fig = go.Figure()
            thresholds = np.linspace(0, 1, 20)
            for metric in metrics:
                if not all_results[metric].empty:
                    ratios = [
                        (all_results[metric]['similarity_score'] >= threshold).mean()
                        for threshold in thresholds
                    ]
                    fig.add_trace(go.Scatter(
                        x=thresholds,
                        y=ratios,
                        name=metric.capitalize(),
                        mode='lines+markers'
                    ))
            fig.update_layout(
                title="Document Ratio vs Threshold",
                xaxis_title="Threshold",
                yaxis_title="Ratio of Documents"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Correlation Analysis
        st.write("#### Metric Correlation Analysis")
        if all(not results.empty for results in all_results.values()):
            correlation_data = pd.DataFrame({
                metric: all_results[metric]['similarity_score']
                for metric in metrics
            })
            correlation_matrix = correlation_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=metrics,
                y=metrics,
                text=np.round(correlation_matrix, 3),
                texttemplate='%{text}',
                textfont={"size": 14},
                colorscale='RdBu'
            ))
            fig.update_layout(title="Correlation Between Metrics")
            st.plotly_chart(fig, use_container_width=True)

        # Detailed Statistics
        st.subheader("3. Detailed Statistics")
        
        # Create detailed statistics table
        detailed_stats = {}
        for metric in metrics:
            if not all_results[metric].empty:
                data = all_results[metric]['similarity_score']
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
                detailed_stats[metric] = stats
        
        if detailed_stats:
            st.table(pd.DataFrame(detailed_stats).round(4))

        # Export Options
        st.subheader("4. Export Options")
        
        # Prepare Excel report
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Summary sheet
            pd.DataFrame(detailed_stats).round(4).to_excel(writer, sheet_name='Summary_Stats')
            
            # Individual metric sheets
            for metric in metrics:
                if not all_results[metric].empty:
                    all_results[metric].round(4).to_excel(
                        writer, 
                        sheet_name=f'{metric}_details'
                    )
            
            # Add batch statistics
            batch_stats.to_excel(writer, sheet_name='Batch_History')

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Full Report (Excel)",
                data=buffer.getvalue(),
                file_name="similarity_analysis_report.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        with col2:
            # Generate summary text report
            summary_text = self.generate_summary_text(detailed_stats, correlation_matrix)
            st.download_button(
                label="Download Summary (Text)",
                data=summary_text,
                file_name="summary_report.txt",
                mime="text/plain"
            )
    def show_classification_metrics(self, threshold: float, metric: str):
        try:
            # Get processing status
            status = self.db.get_processing_status()
            total_all_time = self.db.get_total_processed_count()
            current_results = self.db.query_by_similarity(metric=metric, latest_only=True)
            batch_stats = self.db.get_batch_stats()

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
            if not batch_stats.empty:
                st.metric("Total Batches", len(batch_stats))
                st.metric("Last Batch Size", batch_stats.iloc[0]['doc_count'])
                st.metric("Last Batch Date", batch_stats.iloc[0]['batch_end'].strftime('%Y-%m-%d %H:%M'))
            
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
            # Load true set first to get total rows if not already set
            true_df = self.loader.load_true_set()
            
            status = self.db.get_processing_status()
            
            # If total_rows is 0, set it to the number of rows in the new texts
            if status['total_rows'] == 0:
                new_df = self.loader.load_new_texts(batch_size, resume=False)
                total_rows = len(new_df)
                
                # Update processing status with total rows
                self.db.update_processing_status(0, total_rows, 'in_progress')
                
                # Refresh status
                status = self.db.get_processing_status()
            
            progress_bar = st.progress(0)
            st.write(f"Starting from position: {status['offset']}/{status['total_rows']}")
            
            cleaned_true = self.cleaner.batch_process(true_df['text'].tolist())
            true_embeddings = self.generator.batch_generate(
                cleaned_true,
                true_df['lens_id'].tolist(),
                is_true_set=True
            )
            
            try:
                new_df = self.loader.load_new_texts(batch_size, resume=True)
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
                    
                    # Update progress
                    new_offset = status['offset'] + len(new_df)
                    progress = min(new_offset / max(status['total_rows'], 1), 1.0)
                    progress_bar.progress(progress)
                    
                    st.success(f"Processed {len(new_df)} documents ({new_offset}/{status['total_rows']} total)")
                    
                    # Update processing status
                    if new_offset >= status['total_rows']:
                        self.db.update_processing_status(
                            new_offset,
                            status['total_rows'],
                            'completed'
                        )
                else:
                    st.info("All documents processed!")
                    self.db.update_processing_status(
                        status['total_rows'],
                        status['total_rows'],
                        'completed'
                    )
                    
            except Exception as e:
                # Save last successful position
                self.db.update_processing_status(
                    status['offset'],
                    status['total_rows'],
                    'failed'
                )
                raise e
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Detailed Traceback:")
            st.code(traceback.format_exc())


    def show_results_table(self, threshold: float, metric: str):
        """Display results table"""
        try:
            # Get latest results for each document
            results = self.db.query_by_similarity(metric=metric, latest_only=True)
            
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


    def generate_summary_text(self, detailed_stats: Dict, correlation_matrix: pd.DataFrame) -> str:
        """Generate summary text report"""
        summary = ["SIMILARITY ANALYSIS SUMMARY REPORT", "=" * 30, ""]
        
        # Add timestamp
        summary.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add statistics for each metric
        summary.append("DETAILED STATISTICS")
        summary.append("-" * 20)
        for metric, stats in detailed_stats.items():
            summary.append(f"\n{metric.upper()} SIMILARITY:")
            for stat_name, value in stats.items():
                summary.append(f"{stat_name}: {value:.4f}")
        
        # Add correlation information
        summary.append("\nMETRIC CORRELATIONS")
        summary.append("-" * 20)
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                if i < j:  # Only print upper triangle
                    summary.append(f"{i.capitalize()} vs {j.capitalize()}: {correlation_matrix.loc[i,j]:.4f}")
        
        # Add key findings
        summary.append("\nKEY FINDINGS")
        summary.append("-" * 20)
        for metric, stats in detailed_stats.items():
            summary.append(f"\n{metric.capitalize()} Similarity:")
            summary.append(f"- Average similarity: {stats['Mean']:.4f}")
            summary.append(f"- {stats['Above 0.8']*100:.1f}% of documents above 0.8 threshold")
            summary.append(f"- {stats['Below 0.2']*100:.1f}% of documents below 0.2 threshold")
            
            # Add distribution characterization
            if stats['Skewness'] > 0:
                skew_desc = "positively skewed (more low values)"
            else:
                skew_desc = "negatively skewed (more high values)"
            summary.append(f"- Distribution is {skew_desc}")
            
            # Add spread characterization
            spread = stats['75th Percentile'] - stats['25th Percentile']
            summary.append(f"- Interquartile range: {spread:.4f}")

        # Add processing information
        summary.append("\nPROCESSING INFORMATION")
        summary.append("-" * 20)
        true_embeddings, _ = self.similarity.get_true_embeddings_with_ids()
        new_embeddings, _ = self.similarity.get_new_embeddings_with_ids()
        summary.append(f"Reference Documents: {len(true_embeddings)}")
        summary.append(f"New Documents: {len(new_embeddings)}")
        summary.append(f"Total Comparisons: {len(true_embeddings) * len(new_embeddings)}")
        
        return "\n".join(summary)


def main():
   app = DashboardApp()
   app.main()

if __name__ == "__main__":
   main()