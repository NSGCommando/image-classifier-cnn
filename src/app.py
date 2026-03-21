from fastapi import FastAPI, UploadFile, HTTPException, File
from src.scripts.inference import infer_one
from src.utils.util_functions import load_saved_model
from src.schemas.response_models import PredictResponse
model = load_saved_model()

app = FastAPI()

@app.post("/api/predict-one",response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename:raise HTTPException(status_code=400,detail="Uploaded File has incorrect headers")
    if not file.content_type:raise HTTPException(status_code=400,detail="Uploaded File has no Filename")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    image_bytes = await file.read()
    results = infer_one(image_bytes,model)
    return PredictResponse(
        filename=file.filename,
        predicted_class=results["predicted_class"],
        confidence=results["confidence"]
    )

@app.get("/")
async def root():
    return {"message": "FastAPI Server Running"}