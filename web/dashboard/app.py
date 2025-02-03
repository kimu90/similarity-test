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

    def show_results_table(self, threshold: float, metric: str):
        """Display results table"""
        try:
            results = self.db.query_by_similarity(metric=metric)
            if not results.empty:
                # Add classification column
                results['is_similar'] = results['similarity_score'] >= threshold
                
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

    def show_comprehensive_report(self):
        correlation_matrix = self.compute_correlation_matrix()  # Make sure this method exists

        """Generate comprehensive analysis report"""
        st.header("Comprehensive Similarity Analysis Report")

        # Document Statistics Section
        st.subheader("1. Document Statistics")
        col1, col2, col3 = st.columns(3)
        
        true_embeddings, true_ids = self.similarity.get_true_embeddings_with_ids()
        new_embeddings, new_ids = self.similarity.get_new_embeddings_with_ids()
        
        with col1:
            st.metric("Reference Documents", len(true_embeddings))
            st.metric("New Documents", len(new_embeddings))
        with col2:
            st.metric("Embedding Dimension", true_embeddings.shape[1])
            st.metric("Total Comparisons", len(true_embeddings) * len(new_embeddings))
        with col3:
            st.metric("Total Documents", len(true_embeddings) + len(new_embeddings))
            st.metric("Processing Time", f"{time.time() - self.start_time:.2f}s")

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