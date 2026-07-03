import uuid
from fastapi import FastAPI, UploadFile, File
from src.utils.util_functions_inference import image_handler_http
from src.utils.redis_client import get_redis, add_image_job, RESULT_STREAM_NAME, _RedisClientCache

redis_client_dispatcher = get_redis("dispatcher")

# dispatcher to handle image routing to FastAPI
dispatcher = FastAPI()
job_ids = []

@dispatcher.post("/api/dispatcher")
@image_handler_http
async def predict(image_data_list=None, # injected via decorator
                zipname=None, # injected via decorator
                file: UploadFile = File(...)):
    
    if not image_data_list: # will catch empty list []
        return {"error": "Processing failed"}

    for image in image_data_list:
        job_id = str(uuid.uuid4())
        await add_image_job(redis_client_dispatcher,image["filename"],image["content"],job_id)
        job_ids.append(job_id)

    return {
        "zipname": zipname,
        "status": "processing",
        "queued": len(image_data_list)
    }

@dispatcher.get("/api/dispatcher/results")
async def get_results():
    results = {}
    last_msg_id = "0-0" # This is used to start with first message in the stream
    while len(results) < len(job_ids):
        resp = await redis_client_dispatcher.xread(
            streams={RESULT_STREAM_NAME: last_msg_id}, # the streams arg takes a dict of the stream name and the ID to stream
            block=5000,
            count=10
        )

        if not resp:
            continue

        for stream, messages in resp:
            if not isinstance(messages, list):
                continue
            for message_id, fields in messages:
                last_msg_id = message_id # update the last seen msg ID every time we find a valid message

                job_id = fields[b"job_id"].decode()

                if job_id in job_ids:
                    results[job_id] = {
                        "filename": fields[b"filename"].decode(),
                        "predicted_class": fields[b"predicted_class"].decode(),
                        "confidence": float(fields[b"confidence"].decode())
                    }
    return {"results":results}

@dispatcher.get("/")
async def root():
    return {"message": "FastAPI Dispatcher running"}