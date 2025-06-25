import json
import os
from datetime import datetime
from typing import Literal

class FeedbackLogger:
    """Handles logging of fallback queries to feedback.json"""
    
    def __init__(self, filename: str = "feedback.json"):
        self.filename = filename
        self._initialize_file()
    
    def _initialize_file(self):
        """Create feedback file with empty structure if doesn't exist"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump({"fallback_queries": []}, f, ensure_ascii=False, indent=2)
    
    def log_fallback(
        self,
        query: str,
        fallback_type: Literal["non_agricultural", "web_results"],
        additional_context: dict = None
    ):
        """Log a fallback query with context"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "fallback_type": fallback_type,
            "handled": False,
            "context": additional_context or {}
        }
        
        try:
            with open(self.filename, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data["fallback_queries"].append(entry)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
        except Exception as e:
            print(f"Failed to log feedback: {str(e)}")

    def get_unhandled_feedbacks(self):
        """Retrieve all unhandled feedback entries"""
        with open(self.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [item for item in data["fallback_queries"] if not item["handled"]]