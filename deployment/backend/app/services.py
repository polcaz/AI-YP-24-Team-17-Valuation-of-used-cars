# Реализует бизнес-логику (например, обучение моделей, предсказания)

import pandas as pd
import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import os
from deployment.backend.app.preprocessing import preproc
from deployment.backend.app.class_model import FullModel
from deployment.backend.app.preprocessing_x import preproc_x

# In-memory storage for models and data
models = {}
datasets = {}
datasets_prep = {}
experiments = {}
loaded_model = None

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

        return {"message": "CSV dataset uploaded successfully", "df.isnull": df.isnull().sum().to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


def perform_eda():
    if "current" not in datasets:
        raise HTTPException(status_code=404, detail="No dataset uploaded")
    df = datasets["current"]
    return {
        "statistics": df.describe().to_dict(),
    }

def preprocessing_data():
    if "current" not in datasets:
        raise HTTPException(status_code=404, detail="No dataset uploaded")
    df = datasets["current"]
    train, test = preproc(df)
    datasets_prep["current"] = [train, test]
    return {"message": "Preprocessing of the dataset was successful.", "train.isnull": train.isnull().sum().to_dict()}

def train_model(config):
    if config.id in models:
        raise HTTPException(status_code=400, detail=f"Model '{config.id}' already exists")
    if config.ml_model_type == "full":
        model = FullModel(config.hyperparameters)
        # Подготовка данных
        X_train, y_train, X_test, y_test = model.prepare_data(datasets_prep["current"][0], datasets_prep["current"][1])
        # Построение пайплайна
        model.build_pipeline(X_train)
        # Обучение модели
        model.train_model(X_train, y_train)
        # Оценка модели
        r2 = model.evaluate_model(X_test, y_test)
        # Предсказание
        predictions = model.predict(X_test)
        models[config.id] = model
        experiments[config.id] = {"epochs": list(range(10)), "accuracy": [0.8] * 10, "loss": [0.2] * 10}
    elif config.ml_model_type == "poly":
        model = FullModel(config.hyperparameters)
        # Подготовка данных
        X_train, y_train, X_test, y_test = model.prepare_data(datasets_prep["current"][0], datasets_prep["current"][1])
        # Убираем логарифмирование целевой переменной
        y_train = np.exp(y_train)
        y_test = np.exp(y_test)
        # Построение пайплайна
        model.build_pipeline(X_train)
        # Обучение модели
        model.train_model(X_train, y_train)
        # Оценка модели
        r2 = model.evaluate_model(X_test, y_test)
        # Предсказание
        predictions = model.predict(X_test)
        models[config.id] = model
        experiments[config.id] = {"epochs": list(range(10)), "accuracy": [0.8] * 10, "loss": [0.2] * 10}
    elif config.ml_model_type == "ohe":
        model = FullModel(config.hyperparameters)
        # Подготовка данных
        X_train, y_train, X_test, y_test = model.prepare_data(datasets_prep["current"][0], datasets_prep["current"][1])
        # Убираем логарифмирование целевой переменной
        y_train = np.exp(y_train)
        y_test = np.exp(y_test)
        # Построение пайплайна без полиномиальных признаков
        model.build_pipeline_cat(X_train)
        # Обучение модели
        model.train_model(X_train, y_train)
        # Оценка модели
        r2 = model.evaluate_model(X_test, y_test)
        # Предсказание
        predictions = model.predict(X_test)
        models[config.id] = model
        experiments[config.id] = {"epochs": list(range(10)), "accuracy": [0.8] * 10, "loss": [0.2] * 10}
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")


    return {"message": f"Model '{config.id}' trained successfully, r2: {round(r2, 4)}"}

def load_model_endpoint(request):
    global loaded_model
    model_id = request.id

    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found.")

    loaded_model = models[model_id]
    return {"message": f"Model '{model_id}' loaded"}

def unload_model_endpoint():
    global loaded_model
    if not loaded_model:
        raise HTTPException(status_code=400, detail="No model loaded.")

    loaded_model = None
    return {"message": "Model unloaded"}

def compare_experiments(selected_experiments):
    result = []
    for exp in selected_experiments:
        if exp not in experiments:
            raise HTTPException(status_code=404, detail=f"Experiment '{exp}' not found")
        result.append({"name": exp, **experiments[exp]})
    return result

def make_prediction(model_id, data):
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    model = models[model_id]

    try:
        input_data = pd.DataFrame([data])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")
    try:
        predictions = model.predict(preproc_x(input_data))
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

async def predict_items(file):
    global loaded_model
    if not loaded_model:
        raise HTTPException(status_code=400, detail="No model loaded.")
    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())
    # Load the CSV into a DataFrame
    X = pd.read_csv(temp_file_path)
    os.remove(temp_file_path)

    output = X

    X['predict'] = pd.Series(loaded_model.predict(preproc_x(X)))
    output['predict'] = X['predict']
    output.to_csv('predictions.csv', index=False)
    response = FileResponse(path='predictions.csv',
                            media_type='text/csv', filename='predictions.csv')
    return response

def list_models():
    return {"models": list(models.keys())}

def remove_model(model_id: str):
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found.")

    del models[model_id]
    return {"message": f"Model '{model_id}' removed"}

def remove_all_models():
    models.clear()
    return {"message": "All models removed"}