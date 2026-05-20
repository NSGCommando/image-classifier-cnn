from fastapi import FastAPI, File, UploadFile
from src.scripts.inference import infer_one
from src.utils.util_functions_inference import start_inference_session
from src.schemas.response_models import SinglePrediction
# load required sessions
infer_session = start_inference_session()
http_inference = FastAPI()

@http_inference.post("/api/predict-one",response_model=SinglePrediction)
async def predict(image_file:UploadFile=File(...)):
    """Route to predict result for a single image. Returns SinglePrediction response."""
    image_bytes = await image_file.read()
    if not image_bytes: # will catch empty list []
        return {"error": "Processing failed, no image data received"}
    result = infer_one(image_bytes,infer_session)
    filename = image_file.filename
    if not filename: return {"error":"Processing failed, no filename received"}
    return (SinglePrediction(
        filename=filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"]
    ))

@http_inference.get("/")
async def root():
    return {"message": "FastAPI Server Running with ONNX Inference"}