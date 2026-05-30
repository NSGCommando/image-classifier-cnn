import uvicorn, threading
from pathlib import Path
from src.utils.util_functions_inference import default_parser, run_server
from src.scripts.cli import cli_inference

dispatch_server_cmd = ["uvicorn", "src.scripts.dispatcher:dispatcher", "--host", "0.0.0.0", "--port", "8000"]
inference_server_cmd = ["uvicorn", "src.scripts.http_server:http_server", "--host", "0.0.0.0", "--port", "8001"]

if __name__ == "__main__":
    
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
        thread1 = threading.Thread(target=run_server,args=(dispatch_server_cmd,"dispatch_server"), daemon=True)
        thread2 = threading.Thread(target=run_server,args=(inference_server_cmd,"inference_server"), daemon=True)
        thread1.start()
        thread2.start()

        try:
            # Keep main thread alive while servers run
            while thread1.is_alive() or thread2.is_alive():
                thread1.join(timeout=1)
                thread2.join(timeout=1)
        except KeyboardInterrupt:
            print("Main orchestrator caught Ctrl+C, shutting down all servers...")