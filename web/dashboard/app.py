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

class DashboardApp:
    def __init__(self):
        st.set_page_config(page_title="Text Similarity Analysis", layout="wide")
        self.db = DatabaseManager(config)
        self.loader = DataLoader()
        self.cleaner = TextPreprocessor(config)
        self.generator = EmbeddingGenerator(config, self.db)
        self.centroid_calc = CentroidCalculator(config, self.db)
        self.similarity = SimilarityAnalyzer(config, self.db)
        self.classifier = TextClassifier(config, self.db)
        self.metrics = MetricsCalculator()
        self.visualizer = Visualizer()

    def main(self):
        st.title("Text Similarity Analysis Dashboard")

        # Sidebar controls
        st.sidebar.header("Controls")
        selected_metric = st.sidebar.selectbox(
            "Similarity Metric",
            ["cosine", "jaccard"],
            format_func=lambda x: x.capitalize()
        )
        similarity_threshold = st.sidebar.selectbox(
            "Similarity Threshold",
            [0.3,0.4,0.5,0.6,0.7, 0.8, 0.9],
            format_func=lambda x: f"{x:.1f}"
        )
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.5, 0.05
        )
        batch_size = st.sidebar.number_input("Batch Size", 1000, 80000, 25000)
        
        if st.sidebar.button("Load and Process New Data"):
            with st.spinner("Loading and processing data..."):
                self.process_new_data(batch_size, selected_metric)

        main_tab, report_tab = st.tabs(["Analysis Dashboard", "Comprehensive Report"])

        with main_tab:
            col1, col2 = st.columns(2)
            with col1:
                self.show_similarity_distribution(similarity_threshold, selected_metric)
                self.show_classification_metrics(similarity_threshold, selected_metric)
            with col2:
                self.show_top_similar_texts(selected_metric)
                self.show_results_table(similarity_threshold, selected_metric)

        with report_tab:
            self.show_comprehensive_report(similarity_threshold, confidence_threshold)

    def show_comprehensive_report(self, similarity_threshold: float, confidence_threshold: float):
        try:
            batch_stats = self.db.get_batch_stats()
            metrics = ['cosine', 'jaccard']
            
            st.header("Comprehensive Similarity Analysis Reports")
            metrics_data = {}
            
            for metric in metrics:
                st.subheader(f"{metric.capitalize()} Similarity Report")
                results = self.db.query_by_similarity(
                    metric=metric, 
                    min_score=similarity_threshold, 
                    min_confidence=confidence_threshold
                )
                
                if results.empty:
                    st.warning(f"No {metric} similarity results found.")
                    continue

                status = self.db.get_processing_status(metric=metric)
                status['total_rows'] = 443249
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", status['total_rows'])
                    st.metric("Processed Documents", status['total_processed'])
                with col2:
                    progress = (status['total_processed'] / status['total_rows']) * 100
                    st.metric("Progress", f"{progress:.1f}%")
                with col3:
                    st.metric("Remaining Documents", status['total_rows'] - status['total_processed'])
                    similar_docs = len(results[results['similarity_score'] >= similarity_threshold])
                    st.metric("Similar Documents Count", similar_docs)

                # Store similar/different counts for pie chart
                similar_count = len(results[results['similarity_score'] >= similarity_threshold])
                different_count = len(results[results['similarity_score'] < similarity_threshold])
                metrics_data[metric] = {'similar': similar_count, 'different': different_count}
                
                data = results['similarity_score']
                
                col1, col2 = st.columns(2)
                with col1:
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
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    results.to_excel(writer, sheet_name=f'{metric}_details', index=False)
                    pd.DataFrame.from_dict(stats, orient='index', columns=[metric.capitalize()]).to_excel(writer, sheet_name='Summary_Stats')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label=f"Download {metric.capitalize()} Report (Excel)",
                        data=buffer.getvalue(),
                        file_name=f"{metric}_similarity_report_{similarity_threshold}_{confidence_threshold:.2f}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                st.markdown("---")

            # Add pie chart comparison
            st.subheader("Metrics Comparison")
            if metrics_data:
                fig = go.Figure()
                
                for i, metric in enumerate(metrics_data):
                    fig.add_trace(go.Pie(
                        labels=['Similar', 'Different'],
                        values=[metrics_data[metric]['similar'], metrics_data[metric]['different']],
                        name=metric.capitalize(),
                        title=f"{metric.capitalize()} Similarity",
                        domain={'row': 0, 'column': i},
                        textinfo='percent+label',
                        hole=.3,
                        marker=dict(
                            colors=['rgb(67, 160, 71)', 'rgb(211, 47, 47)']
                        )
                    ))
                
                fig.update_layout(
                    title=f"Similarity Comparison (Threshold: {similarity_threshold}, Confidence: {confidence_threshold:.2f})",
                    grid={'rows': 1, 'columns': 2},
                    annotations=[
                        dict(text="Cosine", x=0.18, y=0.5, showarrow=False, font_size=16),
                        dict(text="Jaccard", x=0.82, y=0.5, showarrow=False, font_size=16)
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # Generate comprehensive report
            st.subheader("Complete Report Download")
            try:
                buffer = self.generate_full_report(similarity_threshold, confidence_threshold, metrics_data)
                if buffer:
                    st.download_button(
                        "Download Complete Analysis Report (Excel)",
                        buffer.getvalue(),
                        f"complete_similarity_analysis_{similarity_threshold}_{confidence_threshold:.2f}.xlsx",
                        "application/vnd.ms-excel"
                    )
                else:
                    st.warning("Unable to generate complete report. Please ensure data is available.")
            except Exception as e:
                st.error(f"Error generating complete report: {str(e)}")
                st.error("Detailed error:")
                st.code(traceback.format_exc())
                
            # Combined metrics download
            combined_data = []
            for metric in metrics:
                results = self.db.query_by_similarity(
                    metric=metric, 
                    min_score=similarity_threshold, 
                    min_confidence=confidence_threshold
                )
                if not results.empty:
                    results['metric'] = metric
                    results['is_similar'] = results['similarity_score'] >= similarity_threshold
                    combined_data.append(results)
            
            if combined_data:
                combined_df = pd.concat(combined_data)
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    "Download Combined Metrics Report",
                    csv,
                    f"combined_metrics_report_{similarity_threshold}_{confidence_threshold:.2f}.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"Error in report generation: {str(e)}")
            st.error("Detailed error:")
            st.code(traceback.format_exc())

    def generate_full_report(self, similarity_threshold: float, confidence_threshold: float, metrics_data: dict):
        if not metrics_data:
            return None
            
        buffer = io.BytesIO()
        
        try:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Text Summary Sheet
                summary_worksheet = workbook.add_worksheet('Executive_Summary')
                bold = workbook.add_format({'bold': True})
                wrap = workbook.add_format({'text_wrap': True})
                
                row = 0
                summary_worksheet.write(row, 0, "SIMILARITY ANALYSIS EXECUTIVE SUMMARY", bold)
                row += 2
                
                summary_worksheet.write(row, 0, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                row += 2
                
                summary_worksheet.write(row, 0, "ANALYSIS PARAMETERS", bold)
                row += 1
                summary_worksheet.write(row, 0, f"Similarity Threshold: {similarity_threshold}")
                row += 1
                summary_worksheet.write(row, 0, f"Confidence Threshold: {confidence_threshold}")
                row += 2
                
                # Results Analysis
                for metric in metrics_data:
                    stats_sheet = workbook.add_worksheet(f'{metric}_Analysis')
                    results = self.db.query_by_similarity(metric=metric)
                    if not results.empty:
                        results = results[results['confidence'] >= confidence_threshold]
                        
                        # Basic stats
                        data = results['similarity_score']
                        stats = {
                            'Total Documents': len(results),
                            'Similar Documents': metrics_data[metric]['similar'],
                            'Different Documents': metrics_data[metric]['different'],
                            'Mean Score': data.mean(),
                            'Median Score': data.median(),
                            'Std Dev': data.std(),
                            'Min Score': data.min(),
                            'Max Score': data.max(),
                            '25th Percentile': data.quantile(0.25),
                            '75th Percentile': data.quantile(0.75)
                        }
                        
                        # Write stats
                        r = 0
                        stats_sheet.write(r, 0, f"{metric.capitalize()} Analysis", bold)
                        r += 2
                        for k, v in stats.items():
                            stats_sheet.write(r, 0, k)
                            stats_sheet.write(r, 1, v)
                            r += 1
                        
                        # Write full results
                        results_sheet = workbook.add_worksheet(f'{metric}_Full_Results')
                        results.to_excel(writer, sheet_name=f'{metric}_Full_Results')
                
                # Combined Results
                combined_data = []
                for metric in metrics_data:
                    results = self.db.query_by_similarity(metric=metric)
                    if not results.empty:
                        results = results[results['confidence'] >= confidence_threshold]
                        results['metric'] = metric
                        results['is_similar'] = results['similarity_score'] >= similarity_threshold
                        combined_data.append(results)
                
                if combined_data:
                    combined_df = pd.concat(combined_data)
                    combined_df.to_excel(writer, sheet_name='Combined_Results', index=False)
                
                # Adjust columns width
                for worksheet in workbook.worksheets():
                    worksheet.set_column(0, 0, 30)
                    worksheet.set_column(1, 1, 15)
            
            return buffer
        except Exception as e:
            print(f"Error in report generation: {str(e)}")
            return None
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
                st.metric("Documents Remaining", status['total_rows'] - status['total_processed'])
            with col2:
                if status['total_rows'] > 0:
                    progress = (status['total_processed'] / status['total_rows']) * 100
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
            st.write(f"Starting from position: {status['total_processed']}/{status['total_rows']}")
            
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
                    
                    new_total_processed = status['total_processed'] + len(new_df)
                    progress = min(new_total_processed / max(status['total_rows'], 1), 1.0)
                    progress_bar.progress(progress)
                    
                    st.success(f"Processed {len(new_df)} documents ({new_total_processed}/{status['total_rows']} total)")
                    
                    status_label = 'completed' if new_total_processed >= status['total_rows'] else 'in_progress'
                    self.db.update_processing_status(
                        new_total_processed,
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
                    status['total_processed'],
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
                        'latest_batch_id',
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