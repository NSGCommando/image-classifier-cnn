from src.scripts.inference import infer_one
from src.utils.util_functions_inference import start_inference_session
from src.utils.redis_client import add_image_result, worker
# load required session
infer_session = start_inference_session()

async def predict_job(redis_client,fields):
    """Predict result for a single image job. Returns SinglePrediction response."""
    job_id = fields[b"job_id"].decode()
    task= fields[b"task"].decode()
    if task != "classify":
        raise RuntimeError
    image_bytes = bytes.fromhex(fields[b"image_data"].decode())
    if not image_bytes: # will catch empty list []
        return {"error": "Processing failed, no image data received"}
    result = infer_one(image_bytes,infer_session)
    filename = fields[b"image_filename"].decode()
    if not filename: return {"error":"Processing failed, no filename received"}
    result = {
        "filename":filename,
        "predicted_class":result["predicted_class"],
        "confidence":result["confidence"]
    }
    await add_image_result(redis_client=redis_client,result=result,job_id=job_id)

async def worker_wrapper(redis_client,worker_id):
    """Wrapper Function to pass correct prediction function to the worker thread"""
    await worker(redis_client=redis_client,predict_fn=predict_job,worker_id=worker_id)