# Главный файл для запуска FastAPI-сервиса.

import uvicorn
from fastapi import FastAPI
# from deployment.backend.app.routers import api


app = FastAPI(title="ML Model Management API", version="1.0")

# Include API router
# app.include_router(api.router, prefix="/api/v1", tags=["ML API"])

@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "ML API is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)