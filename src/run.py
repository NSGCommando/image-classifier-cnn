import uvicorn
from src.utils.util_functions import default_parser
from src.cli import cli_inference
# Run this for the CLI mode

if __name__ == "__main__":
    
    args = default_parser()

    if args.image:
        cli_inference(args.image)

    elif args.server:
        print("Running as API Server!")
        uvicorn.run(
            "src.app:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=False
        )