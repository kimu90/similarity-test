# web/dashboard/plots.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd
import numpy as np
from config import config

class Visualizer:
    def plot_distribution(self, 
                         scores: np.ndarray, 
                         threshold: float,
                         title: str) -> go.Figure:
        """
        Create similarity score distribution plot
        
        Args:
            scores: Array of similarity scores
            threshold: Threshold value to show as vertical line
            title: Plot title
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=30,
            name="Scores",
            opacity=0.75,
            marker_color='blue'
        ))
        
        # Add threshold line
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold:.2f})",
            annotation_position="top"
        )
        
        # Add mean line
        mean_score = np.mean(scores)
        fig.add_vline(
            x=mean_score,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Mean ({mean_score:.2f})",
            annotation_position="bottom"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Similarity Score",
            yaxis_title="Count",
            template="plotly_white",
            showlegend=True,
            bargap=0.1
        )
        
        return fig

    def plot_time_series(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """
        Create time series plot of similarities
        
        Args:
            df: DataFrame with similarity results
            metric: Similarity metric being plotted
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Calculate rolling average
        rolling_avg = df['similarity_score'].rolling(window=10).mean()
        
        # Add raw scores
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['similarity_score'],
            mode='markers',
            name='Raw Scores',
            marker=dict(size=6, opacity=0.5),
            showlegend=True
        ))
        
        # Add rolling average
        fig.add_trace(go.Scatter(
            x=df.index,
            y=rolling_avg,
            mode='lines',
            name='10-point Moving Average',
            line=dict(color='red', width=2),
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"{metric.capitalize()} Similarity Over Time",
            xaxis_title="Sample Index",
            yaxis_title="Similarity Score",
            template="plotly_white"
        )
        
        return fig

    def create_similarity_heatmap(self, 
                                similarity_matrix: np.ndarray, 
                                metric: str) -> go.Figure:
        """
        Create similarity heatmap
        
        Args:
            similarity_matrix: Matrix of pairwise similarities
            metric: Name of similarity metric
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            colorscale='Viridis',
            colorbar=dict(
                title=f"{metric.capitalize()} Similarity",
                thickness=20,
                len=0.7
            )
        ))
        
        fig.update_layout(
            title=f"{metric.capitalize()} Similarity Heatmap",
            xaxis_title="True Set Index",
            yaxis_title="New Text Index",
            template="plotly_white",
            height=600
        )
        
        return fig

    def plot_threshold_analysis(self, 
                              scores: np.ndarray, 
                              thresholds: List[float],
                              metric: str) -> go.Figure:
        """
        Create threshold analysis plot
        
        Args:
            scores: Array of similarity scores
            thresholds: List of threshold values to analyze
            metric: Similarity metric being analyzed
        Returns:
            Plotly figure
        """
        similar_ratios = [
            (scores >= threshold).mean()
            for threshold in thresholds
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=similar_ratios,
            mode='lines+markers',
            name='Similar Ratio',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{metric.capitalize()} Similarity Threshold Analysis",
            xaxis_title="Threshold Value",
            yaxis_title="Ratio of Similar Texts",
            template="plotly_white",
            showlegend=True
        )
        
        return fig

    def create_summary_table(self, 
                           df: pd.DataFrame, 
                           metric: str) -> go.Figure:
        """
        Create summary statistics table
        
        Args:
            df: DataFrame with similarity results
            metric: Similarity metric being summarized
        Returns:
            Plotly figure
        """
        stats = {
            'Metric': [metric.capitalize()],
            'Mean': [df['similarity_score'].mean()],
            'Median': [df['similarity_score'].median()],
            'Std Dev': [df['similarity_score'].std()],
            'Min': [df['similarity_score'].min()],
            'Max': [df['similarity_score'].max()],
            'Similar Ratio': [(df['similarity_score'] >= 0.8).mean()]
        }
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(stats.keys()),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=list(stats.values()),
                fill_color='lavender',
                align='left',
                format=[None, '.3f', '.3f', '.3f', '.3f', '.3f', '.2%'],
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title=f"{metric.capitalize()} Similarity Summary Statistics",
            template="plotly_white"
        )
        
        return fig

    def create_confidence_plot(self, 
                             df: pd.DataFrame, 
                             metric: str) -> go.Figure:
        """
        Create scatter plot of confidence vs similarity scores
        
        Args:
            df: DataFrame with similarity results
            metric: Similarity metric being plotted
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['similarity_score'],
            y=df['confidence'],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.6,
                color=df['similarity_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Similarity Score")
            ),
            name='Scores'
        ))
        
        fig.update_layout(
            title=f"{metric.capitalize()} Similarity vs Confidence",
            xaxis_title="Similarity Score",
            yaxis_title="Confidence",
            template="plotly_white",
            showlegend=False
        )
        
        return fig

if __name__ == "__main__":
    # Test visualizations
    visualizer = Visualizer()
    
    # Create test data
    test_scores = np.random.rand(100)
    test_df = pd.DataFrame({
        'similarity_score': test_scores,
        'confidence': np.random.rand(100),
        'label': test_scores >= 0.8
    })
    
    # Create test plots
    dist_plot = visualizer.plot_distribution(
        test_scores, 
        threshold=0.8,
        title="Test Distribution"
    )
    
    time_plot = visualizer.plot_time_series(
        test_df,
        metric='cosine'
    )
    
    heatmap = visualizer.create_similarity_heatmap(
        np.random.rand(10, 10),
        metric='cosine'
    )
    
    threshold_plot = visualizer.plot_threshold_analysis(
        test_scores,
        thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
        metric='cosine'
    )
    
    summary_table = visualizer.create_summary_table(
        test_df,
        metric='cosine'
    )
    
    confidence_plot = visualizer.create_confidence_plot(
        test_df,
        metric='cosine'
    )
    
    # Save test plots (optional)
    dist_plot.write_html("test_distribution.html")
    time_plot.write_html("test_time_series.html")
    heatmap.write_html("test_heatmap.html")
    threshold_plot.write_html("test_threshold.html")
    summary_table.write_html("test_summary.html")
    confidence_plot.write_html("test_confidence.html")