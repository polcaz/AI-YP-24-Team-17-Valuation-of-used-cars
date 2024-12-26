# Главный файл для запуска FastAPI-сервиса.

import uvicorn
from fastapi import FastAPI
from deployment.backend.app.routers import api

# import logging
# from logging.handlers import TimedRotatingFileHandler

# # Configure logging
# log_handler = TimedRotatingFileHandler("logs/app.log", when="midnight", interval=1)
# log_handler.suffix = "%Y%m%d"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[log_handler],
# )
# logger = logging.getLogger(__name__)




app = FastAPI(title="ML Model Management API",
              docs_url="/api/openapi",
              version="1.0")

# Include API router
app.include_router(api.router, prefix="/api/v1", tags=["ML API"])

@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "ML API is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)