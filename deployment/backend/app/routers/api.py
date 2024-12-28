# Содержит API-эндпоинты для работы с ML-моделями

from fastapi import APIRouter, HTTPException, File, UploadFile
from deployment.backend.app.services import (
    upload_csv_dataset,
    perform_eda,
    preprocessing_data,
    train_model,
    load_model_endpoint,
    unload_model_endpoint,
    list_learning_curve,
    make_prediction,
    predict_items,
    list_models,
    remove_model,
    remove_all_models, unload_model_endpoint,
)
from deployment.backend.app.models import (
    DatasetUploadRequest,
    FitRequest,
    LoadRequest,
    LoadResponse,
    UnloadResponse,
    PredictionRequest,
    LearningCurveRequest,
    ModelListResponse,
    RemoveResponse, LearningCurveRequest,
)
router = APIRouter()

@router.post("/dataset/upload")
async def upload_csv_dataset_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    return await upload_csv_dataset(file)

@router.get("/dataset/eda")
def perform_eda_endpoint():
    return perform_eda()

@router.post("/dataset/preprocessing")
def preprocessing_dataset_endpoint():
    return preprocessing_data()

@router.post("/models/fit")
def train_model_endpoint(request: FitRequest):
    return train_model(request.config)

@router.get("/models/list_models", response_model=ModelListResponse)
def list_models_endpoint():
    return list_models()

@router.post("/models/load", response_model=LoadResponse)
def load_model(request: LoadRequest):
    return load_model_endpoint(request)

@router.post("/models/unload", response_model=UnloadResponse)
def unload_model():
    return unload_model_endpoint()

@router.post("/models/predict_items")
async def make_prediction_items_endpoint(file: UploadFile):
    return await predict_items(file)

@router.post("/models/learning_curve")
def learning_curves_endpoint(request: LearningCurveRequest):
    return list_learning_curve(request.id)

@router.post("/models/predict")
def make_prediction_endpoint(request: PredictionRequest):
    return make_prediction(request.id, request.data)

@router.delete("/models/remove/{model_id}", response_model=RemoveResponse)
def remove_model_endpoint(model_id: str):
    return remove_model(model_id)

@router.delete("/models/remove_all", response_model=RemoveResponse)
def remove_all_models_endpoint():
    return remove_all_models()