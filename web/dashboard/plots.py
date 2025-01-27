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
   def plot_distribution(self, scores: np.ndarray) -> go.Figure:
       """
       Create similarity score distribution plot
       Args:
           scores: Array of similarity scores
       Returns:
           Plotly figure
       """
       fig = go.Figure(data=[go.Histogram(
           x=scores,
           nbinsx=30,
           name="Similarity Scores",
           opacity=0.75
       )])
       
       fig.update_layout(
           title="Similarity Score Distribution",
           xaxis_title="Similarity Score",
           yaxis_title="Count",
           template="plotly_white",
           showlegend=True
       )
       
       return fig

   def plot_metrics(self, results: pd.DataFrame) -> Dict[str, go.Figure]:
       """
       Generate all metric plots
       Args:
           results: DataFrame with similarity results
       Returns:
           Dictionary of plotly figures
       """
       return {
           'similarity_over_time': self._time_series_plot(results),
           'confidence_vs_similarity': self._scatter_plot(results),
           'label_distribution': self._pie_chart(results)
       }

   def _time_series_plot(self, df: pd.DataFrame) -> go.Figure:
       """Create time series plot of similarities"""
       fig = go.Figure()
       
       fig.add_trace(go.Scatter(
           x=range(len(df)),
           y=df['similarity_score'].rolling(window=10).mean(),
           mode='lines',
           name='Rolling Average'
       ))
       
       fig.update_layout(
           title="Similarity Scores Trend",
           xaxis_title="Sample Index",
           yaxis_title="Similarity Score",
           template="plotly_white"
       )
       
       return fig

   def _scatter_plot(self, df: pd.DataFrame) -> go.Figure:
       """Create scatter plot of confidence vs similarity"""
       fig = px.scatter(
           df,
           x='similarity_score',
           y='confidence',
           color='label',
           title="Confidence vs Similarity",
           template="plotly_white",
           opacity=0.6
       )
       
       fig.update_traces(marker=dict(size=8))
       
       return fig

   def _pie_chart(self, df: pd.DataFrame) -> go.Figure:
       """Create pie chart of label distribution"""
       labels = ['Similar', 'Different']
       values = [
           (df['similarity_score'] >= 0.8).sum(),
           (df['similarity_score'] < 0.8).sum()
       ]
       
       fig = go.Figure(data=[go.Pie(
           labels=labels,
           values=values,
           hole=.3
       )])
       
       fig.update_layout(
           title="Classification Distribution",
           template="plotly_white"
       )
       
       return fig

   def create_heatmap(self, similarity_matrix: np.ndarray) -> go.Figure:
       """
       Create similarity heatmap
       Args:
           similarity_matrix: Matrix of pairwise similarities
       Returns:
           Plotly figure
       """
       fig = go.Figure(data=go.Heatmap(
           z=similarity_matrix,
           colorscale='Viridis'
       ))
       
       fig.update_layout(
           title="Similarity Heatmap",
           xaxis_title="True Set Index",
           yaxis_title="New Text Index",
           template="plotly_white"
       )
       
       return fig

if __name__ == "__main__":
   # Test visualizations
   from src.preprocessing.loader import DataLoader
   from src.analysis.similarity import SimilarityAnalyzer
   
   loader = DataLoader()
   similarity = SimilarityAnalyzer(config)
   visualizer = Visualizer()
   
   # Load some test data
   df = pd.DataFrame({
       'similarity_score': np.random.rand(100),
       'confidence': np.random.rand(100),
       'label': np.random.choice([True, False], 100)
   })
   
   # Create test plots
   dist_plot = visualizer.plot_distribution(df['similarity_score'].values)
   metric_plots = visualizer.plot_metrics(df)
   heatmap = visualizer.create_heatmap(np.random.rand(20, 20))
   
   # Save plots (optional)
   dist_plot.write_html("distribution.html")
   metric_plots['similarity_over_time'].write_html("time_series.html")
   heatmap.write_html("heatmap.html")