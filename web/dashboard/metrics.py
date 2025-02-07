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
        """Calculate classification metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predictions, average='binary'
        )
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def calculate_roc(self, scores: np.ndarray, truth: List[bool]) -> Dict:
        """Calculate ROC curve data"""
        fpr, tpr, thresholds = roc_curve(truth, scores)
        roc_auc = auc(fpr, tpr)
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc,
            'thresholds': thresholds.tolist()
        }

    def get_summary_stats(self, results: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        stats = {
            'total_processed': len(results),
            'true_ratio': results['label'].mean(),
            'avg_similarity': results['similarity_score'].mean(),
            'std_similarity': results['similarity_score'].std(),
            'median_similarity': results['similarity_score'].median()
        }
        
        quartiles = results['similarity_score'].quantile([0.25, 0.5, 0.75])
        stats.update({
            'q1_similarity': quartiles[0.25],
            'q2_similarity': quartiles[0.50],
            'q3_similarity': quartiles[0.75]
        })
        
        return stats

    def track_metrics(self, new_metrics: Dict):
        """Track metrics over time"""
        self.metrics_history.append({
            'timestamp': pd.Timestamp.now(),
            **new_metrics
        })
        self.logger.info(f"Tracked new metrics: {new_metrics}")

    def get_metrics_trend(self) -> pd.DataFrame:
        """Get metrics history as DataFrame"""
        return pd.DataFrame(self.metrics_history)

    def calculate_threshold_metrics(self, scores: np.ndarray, thresholds: List[float]) -> pd.DataFrame:
        """Calculate metrics for different thresholds"""
        results = []
        for threshold in thresholds:
            predictions = scores >= threshold
            metrics = self.calculate_metrics(predictions, [True] * len(scores))
            results.append({
                'threshold': threshold,
                **metrics
            })
        return pd.DataFrame(results)


def generate_pdf_report(results_data, metrics_data, status_data, similarity_threshold):
    """Generate PDF report with all visualizations and detailed statistics"""
    temp_files = []
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), 
                            rightMargin=50, leftMargin=50, 
                            topMargin=50, bottomMargin=50)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        title = Paragraph(
            f"Comprehensive Analysis Report<br/>"
            f"Similarity Threshold: {similarity_threshold}",
            title_style
        )
        elements.append(title)

        # Add timestamp
        timestamp = Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            styles['Normal']
        )
        elements.append(timestamp)
        elements.append(Spacer(1, 20))

        for metric in metrics_data:
            # Metric section
            elements.append(Paragraph(f"{metric.capitalize()} Similarity Analysis", 
                                   styles['Heading1']))
            elements.append(Spacer(1, 10))

            # Basic metrics table
            status = status_data[metric]
            basic_metrics = [
                ['Metric', 'Value'],
                ['Similar Documents', f"{metrics_data[metric]['similar']:,}"],
                ['Different Documents', f"{metrics_data[metric]['different']:,}"],
                ['Total Processed', f"{status['total_processed']:,}"],
                ['Progress', f"{(status['total_processed'] / status['total_rows'] * 100):.1f}%"]
            ]
            
            metrics_table = Table(basic_metrics, colWidths=[3*inch, 3*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 20))

            try:
                # Add visualizations
                for fig_name, fig in [
                    ('Distribution', results_data[metric]['fig_dist']),
                    ('Threshold', results_data[metric]['fig_threshold'])
                ]:
                    img_bytes = fig.to_image(format="png", engine="kaleido")
                    
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix='.png',
                        delete=False,
                        dir='/tmp'
                    )
                    temp_files.append(temp_file.name)
                    
                    try:
                        temp_file.write(img_bytes)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                        temp_file.close()
                        
                        if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                            img = Image(temp_file.name, width=6*inch, height=4*inch)
                            elements.append(Paragraph(f"{fig_name} Analysis", styles['Heading2']))
                            elements.append(img)
                            elements.append(Spacer(1, 10))
                    except Exception as e:
                        logging.error(f"Error processing {fig_name} figure: {str(e)}")
            except Exception as e:
                logging.error(f"Error adding visualizations for {metric}: {str(e)}")
            
            elements.append(Spacer(1, 20))

        # Build the PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logging.error(f"Error generating PDF report: {str(e)}")
        traceback.print_exc()
        return None
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logging.error(f"Error cleaning up temp file {temp_file}: {str(e)}")


def show_comprehensive_report(self, similarity_threshold: float):
    """Display comprehensive similarity analysis report"""
    try:
        st.header(f"Comprehensive Analysis Report (Similarity Threshold: {similarity_threshold})")
        
        metrics = ['cosine', 'jaccard', 'concatenated-cosine']
        metrics_data = {}
        results_data = {}
        status_data = {}
        
        for metric in metrics:
            st.subheader(f"{metric.capitalize()} Similarity Report")
            
            # Get all results first with min_score=0
            results = self.db.query_by_similarity(
                metric=metric,
                min_score=0.0  # Get all results
            )
            
            # Get processing status
            status = self.db.get_processing_status(metric=metric)
            # Set total rows based on metric
            total_target = 26712 if metric == 'concatenated-cosine' else 443249
            status['total_rows'] = total_target
            status_data[metric] = status
            
            # Only show warning if no results and status is not completed
            if results.empty and status['status'] != 'completed':
                st.warning(f"No {metric} similarity results found.")
                continue

            # Filter results by threshold after getting status
            similar_results = results[results['similarity_score'] >= similarity_threshold]
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", status['total_rows'])
                st.metric("Processed Documents", status['total_processed'])
                progress = (status['total_processed'] / status['total_rows']) * 100
                st.metric("Progress", f"{progress:.1f}%")
            with col2:
                remaining = status['total_rows'] - status['total_processed']
                st.metric("Remaining Documents", remaining)
                similar_docs = len(similar_results)
                st.metric("Similar Documents", similar_docs)
                different_docs = len(results) - similar_docs
                st.metric("Different Documents", different_docs)
            
            # Store metrics data
            metrics_data[metric] = {
                'similar': similar_docs,
                'different': different_docs,
                'results': results
            }
            
            # Create visualizations if we have data
            if not results.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    fig_dist = go.Figure(data=go.Violin(
                        y=results['similarity_score'],
                        name=metric.capitalize(),
                        box_visible=True,
                        meanline_visible=True
                    ))
                    
                    title = ("Concatenated Cosine Distribution" if metric == 'concatenated-cosine' 
                            else f"{metric.capitalize()} Score Distribution")
                    
                    fig_dist.update_layout(
                        title=title,
                        yaxis_title="Similarity Score",
                        height=400
                    )
                    
                    # Add specific range for concatenated-cosine
                    if metric == 'concatenated-cosine':
                        fig_dist.update_layout(
                            yaxis_range=[0, 1],
                            title_text=title + " (0-1 Range)",
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
                    
                    threshold_title = ("Concatenated Cosine Threshold Analysis" 
                                     if metric == 'concatenated-cosine' 
                                     else "Document Ratio vs Threshold")
                    
                    fig_threshold.update_layout(
                        title=threshold_title,
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
                
                # Download raw data
                csv = results.to_csv(index=False)
                st.download_button(
                    label=f"Download {metric.capitalize()} Raw Data (CSV)",
                    data=csv,
                    file_name=f"{metric}_data_{similarity_threshold}.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
        
        # Combined metrics comparison if we have data
        if any(len(data['results']) > 0 for data in metrics_data.values()):
            st.subheader("Metrics Comparison")
            
            fig = go.Figure()
            for i, metric in enumerate(metrics_data):
                if len(metrics_data[metric]['results']) > 0:
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
                title=f"Similarity Comparison (Threshold: {similarity_threshold})",
                grid={'rows': 1, 'columns': len(metrics_data)},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Combined metrics download
            combined_data = []
            for metric, data in metrics_data.items():
                if len(data['results']) > 0:
                    results = data['results'].copy()
                    results['metric'] = metric
                    results['is_similar'] = results['similarity_score'] >= similarity_threshold
                    combined_data.append(results)
            
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                
                # Download combined raw data
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    "Download Combined Raw Data (CSV)",
                    csv,
                    f"combined_metrics_{similarity_threshold}.csv",
                    "text/csv"
                )
                
                # Create summary DataFrame
                summary_data = []
                for metric, data in metrics_data.items():
                    if len(data['results']) > 0:
                        results = data['results']
                        summary_data.append({
                            'Metric': metric,
                            'Total Documents': len(results),
                            'Similar Documents': data['similar'],
                            'Different Documents': data['different'],
                            'Mean Score': results['similarity_score'].mean(),
                            'Median Score': results['similarity_score'].median()
                        })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                st.dataframe(summary_df)
                
                # Download summary as CSV
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    "Download Summary Statistics (CSV)",
                    summary_csv,
                    f"summary_stats_{similarity_threshold}.csv",
                    "text/csv"
                )
                
                # Generate and offer PDF download
                try:
                    pdf_buffer = generate_pdf_report(
                        results_data=results_data,
                        metrics_data=metrics_data,
                        status_data=status_data,
                        similarity_threshold=similarity_threshold
                    )
                    
                    if pdf_buffer:
                        st.download_button(
                            "📄 Download Complete Report (PDF)",
                            pdf_buffer,
                            f"complete_analysis_report_{similarity_threshold}.pdf",
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