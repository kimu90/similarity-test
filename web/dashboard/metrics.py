# web/dashboard/metrics.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import logging
from config import config
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime
import traceback
import os
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


class MetricsCalculator:
    def __init__(self):
        """Initialize metrics calculator"""
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, predictions: List[bool], truth: List[bool]) -> Dict:
        """
        Calculate classification metrics
        Args:
            predictions: Predicted labels
            truth: True labels
        Returns:
            Dictionary of metrics
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predictions, average='binary'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def calculate_roc(self, scores: np.ndarray, truth: List[bool]) -> Dict:
        """
        Calculate ROC curve data
        Args:
            scores: Similarity scores
            truth: True labels
        Returns:
            Dictionary with ROC curve data
        """
        fpr, tpr, thresholds = roc_curve(truth, scores)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc,
            'thresholds': thresholds.tolist()
        }

    def get_summary_stats(self, results: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics
        Args:
            results: DataFrame with similarity results
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'total_processed': len(results),
            'true_ratio': results['label'].mean(),
            'avg_confidence': results['confidence'].mean(),
            'avg_similarity': results['similarity_score'].mean(),
            'std_similarity': results['similarity_score'].std(),
            'median_similarity': results['similarity_score'].median(),
            'threshold_violations': (
                results['confidence'] < results['similarity_score']
            ).sum()
        }
        
        # Add quartile information
        quartiles = results['similarity_score'].quantile([0.25, 0.5, 0.75])
        stats.update({
            'q1_similarity': quartiles[0.25],
            'q2_similarity': quartiles[0.50],
            'q3_similarity': quartiles[0.75]
        })
        
        return stats

    def track_metrics(self, new_metrics: Dict):
        """
        Track metrics over time
        Args:
            new_metrics: New metrics to track
        """
        self.metrics_history.append({
            'timestamp': pd.Timestamp.now(),
            **new_metrics
        })
        self.logger.info(f"Tracked new metrics: {new_metrics}")

    def get_metrics_trend(self) -> pd.DataFrame:
        """
        Get metrics history as DataFrame
        Returns:
            DataFrame with metrics history
        """
        return pd.DataFrame(self.metrics_history)

    def calculate_threshold_metrics(self, 
                                 scores: np.ndarray, 
                                 thresholds: List[float]) -> pd.DataFrame:
        """
        Calculate metrics for different thresholds
        Args:
            scores: Similarity scores
            thresholds: List of thresholds to test
        Returns:
            DataFrame with threshold metrics
        """
        results = []
        for threshold in thresholds:
            predictions = scores >= threshold
            metrics = self.calculate_metrics(predictions, [True] * len(scores))
            results.append({
                'threshold': threshold,
                **metrics
            })
        return pd.DataFrame(results)


def generate_pdf_report(results_data, metrics_data, status_data, similarity_threshold, confidence_threshold):
    """Generate PDF report with all visualizations and data"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        elements = []
        styles = getSampleStyleSheet()

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        title = Paragraph(f"Comprehensive Analysis Report<br/>Similarity: {similarity_threshold}, Confidence: {confidence_threshold}",
                         title_style)
        elements.append(title)

        # Add timestamp
        timestamp = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             styles['Normal'])
        elements.append(timestamp)
        elements.append(Spacer(1, 20))

        for metric in metrics_data:
            # Add metric section
            elements.append(Paragraph(f"{metric.capitalize()} Similarity Analysis", styles['Heading1']))
            elements.append(Spacer(1, 10))

            # Add metrics table
            status = status_data[metric]
            table_data = [
                ['Metric', 'Value'],
                ['Similar Documents', f"{metrics_data[metric]['similar']:,}"],
                ['Different Documents', f"{metrics_data[metric]['different']:,}"],
                ['Total Processed', f"{status['total_processed']:,}"],
                ['Progress', f"{(status['total_processed'] / status['total_rows'] * 100):.1f}%"]
            ]
            
            metrics_table = Table(table_data, colWidths=[2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 20))

            try:
                # Add visualizations
                fig_dist = results_data[metric]['fig_dist']
                img_bytes_dist = fig_dist.to_image(format="png", engine="kaleido")
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_dist:
                    tmp_dist.write(img_bytes_dist)
                    tmp_dist.flush()
                    img_dist = Image(tmp_dist.name, width=4*inch, height=3*inch)
                    elements.append(img_dist)
                    tmp_dist.close()
                    os.unlink(tmp_dist.name)

                fig_threshold = results_data[metric]['fig_threshold']
                img_bytes_threshold = fig_threshold.to_image(format="png", engine="kaleido")
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_threshold:
                    tmp_threshold.write(img_bytes_threshold)
                    tmp_threshold.flush()
                    img_threshold = Image(tmp_threshold.name, width=4*inch, height=3*inch)
                    elements.append(img_threshold)
                    tmp_threshold.close()
                    os.unlink(tmp_threshold.name)
                
            except Exception as e:
                print(f"Error adding visualizations for {metric}: {str(e)}")
            
            elements.append(Spacer(1, 20))

        # Build the PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        traceback.print_exc()
        return None

def show_comprehensive_report(similarity_threshold: float, confidence_threshold: float, db):
    """Display comprehensive similarity analysis report with appropriate download options"""
    try:
        st.header(f"Comprehensive Analysis Report (Similarity: {similarity_threshold}, Confidence: {confidence_threshold})")
        
        metrics = ['cosine', 'jaccard']
        metrics_data = {}
        status_data = {}
        results_data = {}
        
        for metric in metrics:
            st.subheader(f"{metric.capitalize()} Similarity Report")
            
            # Get results for current metric
            results = db.query_by_similarity(
                metric=metric,
                min_score=similarity_threshold,
                min_confidence=confidence_threshold
            )
            
            if results.empty:
                st.warning(f"No {metric} similarity results found.")
                continue

            # Get processing status
            status = db.get_processing_status(metric=metric)
            status['total_rows'] = 443249  # Total target documents
            status_data[metric] = status
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", status['total_rows'])
                st.metric("Processed Documents", status['total_processed'])
            with col2:
                progress = (status['total_processed'] / status['total_rows']) * 100
                st.metric("Progress", f"{progress:.1f}%")
            with col3:
                remaining = status['total_rows'] - status['total_processed']
                st.metric("Remaining Documents", remaining)
                similar_docs = len(results[results['similarity_score'] >= similarity_threshold])
                st.metric("Similar Documents", similar_docs)
            
            # Store metrics data for later use
            metrics_data[metric] = {
                'similar': len(results[results['similarity_score'] >= similarity_threshold]),
                'different': len(results[results['similarity_score'] < similarity_threshold])
            }
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution plot
                fig_dist = go.Figure(data=go.Violin(
                    y=results['similarity_score'],
                    name=metric.capitalize(),
                    box_visible=True,
                    meanline_visible=True
                ))
                fig_dist.update_layout(
                    title=f"{metric.capitalize()} Score Distribution",
                    yaxis_title="Similarity Score",
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Threshold analysis plot
                thresholds = np.linspace(0, 1, 20)
                ratios = [(results['similarity_score'] >= threshold).mean() 
                         for threshold in thresholds]
                
                fig_threshold = go.Figure(data=go.Scatter(
                    x=thresholds,
                    y=ratios,
                    mode='lines+markers',
                    name=metric.capitalize()
                ))
                fig_threshold.update_layout(
                    title="Document Ratio vs Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="Ratio of Documents",
                    height=400
                )
                st.plotly_chart(fig_threshold, use_container_width=True)
                
                # Store figures for PDF
                results_data[metric] = {
                    'fig_dist': fig_dist,
                    'fig_threshold': fig_threshold
                }
            
            # Detailed statistics
            st.subheader("Detailed Statistics")
            stats = {
                'Mean': results['similarity_score'].mean(),
                'Median': results['similarity_score'].median(),
                'Std Dev': results['similarity_score'].std(),
                'Min': results['similarity_score'].min(),
                'Max': results['similarity_score'].max(),
                '25th Percentile': results['similarity_score'].quantile(0.25),
                '75th Percentile': results['similarity_score'].quantile(0.75),
                'Above 0.8': (results['similarity_score'] >= 0.8).mean(),
                'Below 0.2': (results['similarity_score'] < 0.2).mean()
            }
            
            st.table(pd.DataFrame.from_dict(stats, orient='index', 
                                          columns=[f"{metric.capitalize()} Value"]))
            
            # Download raw data as CSV
            csv = results.to_csv(index=False)
            st.download_button(
                label=f"Download {metric.capitalize()} Raw Data (CSV)",
                data=csv,
                file_name=f"{metric}_data_{similarity_threshold}_{confidence_threshold}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
        
        # Combined metrics comparison
        if metrics_data:
            st.subheader("Metrics Comparison")
            
            fig = go.Figure()
            for i, metric in enumerate(metrics_data):
                fig.add_trace(go.Pie(
                    labels=['Similar', 'Different'],
                    values=[metrics_data[metric]['similar'], 
                           metrics_data[metric]['different']],
                    name=metric.capitalize(),
                    domain={'row': 0, 'column': i},
                    title=f"{metric.capitalize()} Similarity",
                    textinfo='percent+label',
                    hole=.3
                ))
            
            fig.update_layout(
                title=f"Similarity Comparison (Threshold: {similarity_threshold}, "
                      f"Confidence: {confidence_threshold})",
                grid={'rows': 1, 'columns': 2},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Combined metrics download
            combined_data = []
            for metric in metrics:
                results = db.query_by_similarity(
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
                
                # Download combined raw data
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    "Download Combined Raw Data (CSV)",
                    csv,
                    f"combined_metrics_{similarity_threshold}_{confidence_threshold}.csv",
                    "text/csv"
                )
                
                # Create and download summary statistics
                summary_df = pd.DataFrame({
                    'Metric': [],
                    'Total Documents': [],
                    'Similar Documents': [],
                    'Different Documents': [],
                    'Mean Score': [],
                    'Median Score': []
                })
                
                for metric in metrics:
                    metric_data = combined_df[combined_df['metric'] == metric]
                    summary_df = pd.concat([summary_df, pd.DataFrame({
                        'Metric': [metric],
                        'Total Documents': [len(metric_data)],
                        'Similar Documents': [len(metric_data[metric_data['is_similar']])],
                        'Different Documents': [len(metric_data[~metric_data['is_similar']])],
                        'Mean Score': [metric_data['similarity_score'].mean()],
                        'Median Score': [metric_data['similarity_score'].median()]
                    })])
                
                # Download summary as CSV
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    "Download Summary Statistics (CSV)",
                    summary_csv,
                    f"summary_stats_{similarity_threshold}_{confidence_threshold}.csv",
                    "text/csv"
                )
                
                # Generate and offer PDF download
                try:
                    pdf_buffer = generate_pdf_report(
                        results_data=results_data,
                        metrics_data=metrics_data,
                        status_data=status_data,
                        similarity_threshold=similarity_threshold,
                        confidence_threshold=confidence_threshold
                    )
                    
                    if pdf_buffer:
                        st.download_button(
                            "📄 Download Complete Report (PDF)",
                            pdf_buffer,
                            f"complete_analysis_report_{similarity_threshold}_{confidence_threshold}.pdf",
                            "application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.error("Detailed error:")
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.error("Detailed error:")
        st.code(traceback.format_exc())