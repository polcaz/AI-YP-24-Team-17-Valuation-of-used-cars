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

class LoadRequest(BaseModel):
    id: str

class LoadResponse(BaseModel):
    message: str = Field(..., example="Model 'model_name' loaded")

class UnloadResponse(BaseModel):
    message: str = Field(..., example="Model 'model_name' unloaded")

class PredictionRequest(BaseModel):
    id: str
    data: dict

class PredictionResponse(BaseModel):
    predictions: List[float]

class LearningCurveRequest(BaseModel):
    id: str

class LearningCurveResponse(BaseModel):
    data_sizes: List[float]
    train_errors: List[float]
    test_errors: List[float]

class ModelListResponse(BaseModel):
    models: List[str]

class RemoveResponse(BaseModel):
    message: str = Field(..., example="Model 'model_name' removed")