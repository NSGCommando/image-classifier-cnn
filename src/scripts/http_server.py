from fastapi import FastAPI, UploadFile, File
from src.scripts.inference import infer_one
from src.utils.util_functions import load_saved_model, image_handler_http
from src.schemas.response_models import PredictResponse, SinglePrediction
model = load_saved_model()

http_server = FastAPI()

@http_server.post("/api/predict-one",response_model=PredictResponse)
@image_handler_http
async def predict(image_data_list=None, # injected via decorator
                zipname=None, # injected via decorator
                file: UploadFile = File(...)):
    results = []
    if not image_data_list: # will catch empty list []
        return {"error": "Processing failed"}
    for image in image_data_list:
        result = infer_one(image["content"],model)
        filename = image["filename"]
        results.append(SinglePrediction(
            filename=filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"]
        ))
    return PredictResponse(
        zipname=zipname,
        results=results
    )

@http_server.get("/")
async def root():
    return {"message": "FastAPI Server Running"}