import httpx
import asyncio
from fastapi import FastAPI, UploadFile, File
from src.utils.util_functions_inference import image_handler_http
from src.schemas.response_models import PredictResponse
from src.utils.constants import PortAddresses

# dispatcher to handle image routing to FastAPI
dispatcher = FastAPI()
inference_url = f"{PortAddresses.http_inferer.value}/api/predict-one"

@dispatcher.post("/api/dispatcher",response_model=PredictResponse)
@image_handler_http
async def predict(image_data_list=None, # injected via decorator
                zipname=None, # injected via decorator
                file: UploadFile = File(...)):
    results = []
    if not image_data_list: # will catch empty list []
        return {"error": "Processing failed"}
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                url=inference_url,
                # The fieldname for the file MUST be the same as the argument Inference API expects
                files={"image_file":(image["filename"],image["content"])},
            ) for image in image_data_list
        ]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            if resp.status_code == 200:
                results.append(resp.json())
            else:
                print("Inference server error:", resp.status_code, resp.text)
    return PredictResponse(
        zipname=zipname,
        results=results
    )

@dispatcher.get("/")
async def root():
    return {"message": "FastAPI Dispatcher running"}