import threading, asyncio, logging
from pathlib import Path
from src.utils.util_functions_inference import default_parser, run_server
from src.scripts.cli import cli_inference
from src.scripts.inference_worker import worker_wrapper
from src.utils.redis_client import create_worker_group, get_redis
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

dispatch_server_cmd = ["uvicorn", "src.scripts.dispatcher:dispatcher", "--host", "0.0.0.0", "--port", "8000"]
# inference_server_cmd = ["uvicorn", "src.scripts.http_server:http_server", "--host", "0.0.0.0", "--port", "8001"]

async def start_workers(redis_client,worker_count:int):
    tasks = [
        asyncio.create_task(worker_wrapper(redis_client=redis_client,worker_id=i))
        for i in range(worker_count)
    ]
    await asyncio.gather(*tasks)

async def run_app():
    args = default_parser()

    if args.file:
        print("Running as file inference!")
        user_path = Path(args.file)
        if not user_path.is_file():raise TypeError("Argument for --file flag must be a path to an image or zip")
        cli_inference(filepath=user_path,filetype="file")
    
    elif args.folder:
        print("Running as local directory inference!")
        user_path = Path(args.folder)
        if user_path.is_file():raise TypeError("Argument for --folder flag must be a folder/directory path")
        cli_inference(filepath=user_path,filetype="folder")

    elif args.server:
        # start dispatcher thread
        thread1 = threading.Thread(target=run_server,args=(dispatch_server_cmd,"dispatch_server"), daemon=True)
        thread1.start()
        logger.info("Dispatcher Startup Checkpoint")
            
        # create redis worker tasks and the redis worker group
        redis_client_orchestrator = await get_redis("orchestrator")
        await create_worker_group(redis_client=redis_client_orchestrator)
        await start_workers(redis_client_orchestrator,int(args.server))


if __name__ == "__main__":
    asyncio.run(run_app())