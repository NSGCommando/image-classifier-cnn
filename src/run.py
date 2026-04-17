import uvicorn
from pathlib import Path
from src.utils.util_functions_inference import default_parser
from src.scripts.cli import cli_inference
# Run this for the CLI mode

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
        print("Running as API Server!")
        uvicorn.run(
            "src.scripts.http_server:http_server", 
            host="0.0.0.0", 
            port=8000, 
            reload=False
        )