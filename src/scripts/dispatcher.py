import uuid
from fastapi import FastAPI, UploadFile, File
from src.schemas.response_models import SinglePrediction, PredictResponse
from src.utils.util_functions_inference import image_handler_http
from src.utils.redis_client import get_redis, add_image_job, _RedisClientCache
from src.utils.constants import redis_labels

RESULT_STREAM_NAME = redis_labels.RESULT_STREAM_NAME.value
redis_client_dispatcher = get_redis("dispatcher")

# dispatcher to handle image routing to FastAPI
dispatcher = FastAPI()

@dispatcher.post("/api/dispatcher")
@image_handler_http
async def predict(image_data_list=None, # injected via decorator
                zipname=None, # injected via decorator
                file: UploadFile = File(...)):
 
    if not image_data_list: # will catch empty list []
        return {"error": "Processing failed"}

    # create a random UUID for the batch of images
    batch_id = str(uuid.uuid4())

    for image in image_data_list:
        job_id = str(uuid.uuid4())
        await add_image_job(redis_client_dispatcher,image["filename"],image["content"],job_id, batch_id)

    # store metadata for the batch
    await redis_client_dispatcher.hset(
        name=f"batch:{batch_id}",
        mapping={
            "zipname":zipname,
            "expected_jobs":len(image_data_list),
            "status":"processing"
        }
    )

    await redis_client_dispatcher.expire(f"batch:{batch_id}", 120) # batch metadata expires after 2 minutes
    return {
        "batch": batch_id,
        "status": "processing",
        "queued": len(image_data_list)
    }

@dispatcher.get("/api/dispatcher/results/{batch_id}", response_model=PredictResponse)
async def get_results(batch_id:str):
    """
    API endpoint for retrieval of batch results.
    Input Param: batch_id - The ID of the batch to retrieve results for.
    Returns a PredictResponse object with all results and the zipname.
    """
    results = []
    last_msg_id = "0-0" # This is used to start with first message in the stream
    batch = await redis_client_dispatcher.hgetall(f"batch:{batch_id}") # returns BYTES, decoding necessary
    zipname = batch[b"zipname"].decode()
    jobs = int(batch[b"expected_jobs"])
    while len(results) < jobs:
        resp = await redis_client_dispatcher.xread(
            streams={f"{RESULT_STREAM_NAME}-{batch_id}": last_msg_id}, # the streams arg takes a dict of the stream name and the ID to stream
            block=5000,
        )

        if not resp:
            continue

        for stream, messages in resp:
            if not isinstance(messages, list):
                continue
            for message_id, fields in messages:
                last_msg_id = message_id # update the last seen msg ID every time we find a valid message

                results.append(SinglePrediction(
                    filename=fields[b"filename"].decode(),
                    predicted_class=fields[b"predicted_class"].decode(),
                    confidence=float(fields[b"confidence"].decode())
                ))
    final_result = PredictResponse(
        zipname = zipname,
        results = results
    )
    return final_result

@dispatcher.get("/")
async def root():
    return {"message": "FastAPI Dispatcher running"}