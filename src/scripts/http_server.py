from fastapi import FastAPI, UploadFile, File
from src.scripts.inference import infer_one
from src.utils.util_functions_inference import image_handler_http, start_inference_session
from src.schemas.response_models import PredictResponse, SinglePrediction
# load required sessions
infer_session = start_inference_session()
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
        result = infer_one(image["content"],infer_session)
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
    return {"message": "FastAPI Server Running with ONNX Inference"}