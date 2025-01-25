# web/dashboard/plots.py

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd
import numpy as np

class Visualizer:
   def plot_distribution(self, scores: np.ndarray) -> go.Figure:
       """Create similarity score distribution plot"""
       fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=30)])
       fig.update_layout(
           title="Similarity Score Distribution",
           xaxis_title="Similarity Score",
           yaxis_title="Count"
       )
       return fig

   def plot_metrics(self, results: pd.DataFrame) -> Dict[str, go.Figure]:
       """Generate all metric plots"""
       return {
           'similarity_over_time': self._time_series_plot(results),
           'confidence_vs_similarity': self._scatter_plot(results),
           'label_distribution': self._pie_chart(results)
       }

   def _time_series_plot(self, df: pd.DataFrame) -> go.Figure:
       fig = px.line(df, x='created_at', y='similarity_score')
       fig.update_layout(title="Similarity Scores Over Time")
       return fig

   def _scatter_plot(self, df: pd.DataFrame) -> go.Figure:
       fig = px.scatter(df, x='similarity_score', y='confidence',
                       color='label', marginal_y='violin')
       fig.update_layout(title="Confidence vs Similarity")
       return fig

   def _pie_chart(self, df: pd.DataFrame) -> go.Figure:
       labels = ['TRUE', 'FALSE']
       values = [df['label'].sum(), len(df) - df['label'].sum()]
       fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
       fig.update_layout(title="Classification Distribution")
       return fig