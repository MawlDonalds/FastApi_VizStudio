# models.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PromptRequest(BaseModel):
    promt : str

class VisualizationData(BaseModel):
    chart_type: str
    data: Dict[str, Any]
    layout: Dict[str, Any]
    title: str

class BIResponse(BaseModel):
    success: bool
    sql_query: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    visualizations: Optional[List[VisualizationData]] = None
    message: str
    debug_info: Optional[Dict[str, Any]] = None