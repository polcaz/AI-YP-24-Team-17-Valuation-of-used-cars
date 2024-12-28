# Определяет схемы запросов и ответов (использует Pydantic)

from typing import List, Dict, Any
from pydantic import BaseModel, Field

class DatasetUploadRequest(BaseModel):
    data: List[Dict[str, Any]]

class ModelConfig(BaseModel):
    id: str
    ml_model_type: str
    hyperparameters: float

class FitRequest(BaseModel):
    config: ModelConfig

class FitResponse(BaseModel):
    message: str

class ModelListResponse(BaseModel):
    models: List[str]

class PredictionRequest(BaseModel):
    id: str
    X: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]

class ExperimentComparisonRequest(BaseModel):
    experiments: List[str]