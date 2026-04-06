import uvicorn
from pathlib import Path
from src.utils.util_functions import default_parser
from src.scripts.cli import cli_inference
# Run this for the CLI mode

if __name__ == "__main__":
    
    args = default_parser()

    if args.image:
        user_path = Path(args.image)
        cli_inference(user_path)

    elif args.server:
        print("Running as API Server!")
        uvicorn.run(
            "src.scripts.http_server:http_server", 
            host="0.0.0.0", 
            port=8000, 
            reload=False
        )