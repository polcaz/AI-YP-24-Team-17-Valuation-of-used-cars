# Содержит API-эндпоинты для работы с ML-моделями

from fastapi import APIRouter, HTTPException, File, UploadFile
from deployment.backend.app.services import (
    upload_csv_dataset,
    perform_eda,
    preprocessing_data,
    train_model,
    compare_experiments,
    make_prediction,
)
from deployment.backend.app.models import (
    DatasetUploadRequest,
    FitRequest,
    PredictionRequest,
    ExperimentComparisonRequest,
)
router = APIRouter()

@router.post("/dataset/upload")
async def upload_csv_dataset_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    return await upload_csv_dataset(file)

@router.get("/eda")
def perform_eda_endpoint():
    return perform_eda()

@router.post("/preprocessing")
def preprocessing_dataset_endpoint():
    return preprocessing_data()

@router.post("/models/fit")
def train_model_endpoint(request: FitRequest):
    return train_model(request.config)

@router.post("/experiments/compare")
def compare_experiments_endpoint(request: ExperimentComparisonRequest):
    return compare_experiments(request.experiments)

@router.post("/models/predict")
def make_prediction_endpoint(request: PredictionRequest):
    return make_prediction(request.id, request.X)