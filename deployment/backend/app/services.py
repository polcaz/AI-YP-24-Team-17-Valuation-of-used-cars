# Реализует бизнес-логику (например, обучение моделей, предсказания)

import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LinearRegression, LogisticRegression
import os
from deployment.backend.app.preprocessing import preproc

# In-memory storage for models and data
models = {}
datasets = {}
datasets_prep = {}
experiments = {}

async def upload_csv_dataset(file):
    try:
        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Load the CSV into a DataFrame
        df = pd.read_csv(temp_file_path)
        datasets["current"] = df
        os.remove(temp_file_path)

        return {"message": "CSV dataset uploaded successfully", "columns": list(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


async def perform_eda():
    if "current" not in datasets:
        raise HTTPException(status_code=404, detail="No dataset uploaded")
    df = datasets["current"]
    return {
        "statistics": df.describe().to_dict(),
        # "info": df.info().to_dict(),
        # "correlation": df.corr().to_dict(),
    }

def preprocessing_data():
    if "current" not in datasets:
        raise HTTPException(status_code=404, detail="No dataset uploaded")
    df = datasets["current"]
    train, test = preproc(df)
    datasets_prep["current"] = [train, test]
    return {"message": "Preprocessing of the dataset was successful.", "train": train.describe().to_dict()}

def train_model(config):
    if config.id in models:
        raise HTTPException(status_code=400, detail=f"Model '{config.id}' already exists")
    if config.ml_model_type == "linear":
        model = LinearRegression(**config.hyperparameters)
    elif config.ml_model_type == "logistic":
        model = LogisticRegression(**config.hyperparameters)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")
    # Dummy data for demonstration
    X = [[i] for i in range(10)]
    y = [i * 2 for i in range(10)]
    model.fit(X, y)
    models[config.id] = model
    experiments[config.id] = {"epochs": list(range(10)), "accuracy": [0.8] * 10, "loss": [0.2] * 10}
    return {"message": f"Model '{config.id}' trained successfully"}

def compare_experiments(selected_experiments):
    result = []
    for exp in selected_experiments:
        if exp not in experiments:
            raise HTTPException(status_code=404, detail=f"Experiment '{exp}' not found")
        result.append({"name": exp, **experiments[exp]})
    return result

def make_prediction(model_id, X):
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    predictions = models[model_id].predict(X)
    return {"predictions": predictions.tolist()}