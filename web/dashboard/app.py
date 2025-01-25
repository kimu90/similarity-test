# web/dashboard/app.py

from fastapi import FastAPI, HTTPException
from typing import Dict, List
from src.storage.db_manager import DatabaseManager
from src.analysis.classifier import TextClassifier
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

class DashboardApp:
   def __init__(self, config: Dict):
       self.db = DatabaseManager(config['database'])
       self.classifier = TextClassifier(config)
       self.logger = logging.getLogger(__name__)

   @app.get("/data")
   async def get_data(self, filters: Dict) -> Dict:
       """Get filtered classification results"""
       try:
           df = self.db.query_by_similarity(
               min_score=filters.get('min_similarity', 0),
               max_score=filters.get('max_similarity', 1)
           )
           return {
               'results': df.to_dict('records'),
               'metrics': self._calculate_metrics(df)
           }
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   @app.post("/threshold")
   async def update_threshold(self, threshold: float):
       """Update classification threshold"""
       try:
           self.classifier.threshold = threshold
           # Reclassify existing results
           results = self.db.query_by_similarity()
           new_classifications = self.classifier.batch_classify(
               results['similarity_score'].values,
               results['text_id'].values
           )
           self.db.store_results(new_classifications)
           return {"status": "success"}
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   def _calculate_metrics(self, df) -> Dict:
       """Calculate current performance metrics"""
       return {
           'total': len(df),
           'true_count': df['label'].sum(),
           'average_confidence': df['confidence'].mean(),
           'median_similarity': df['similarity_score'].median()
       }

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)