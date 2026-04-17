from pathlib import Path
from src.scripts.inference import infer_one
from src.utils.constants import VALID_EXTENSIONS
from src.utils.util_functions_inference import img_search, start_inference_session, print_results, zip_parser
# Run this for the CLI mode
def cli_inference(filepath:Path, filetype):
    """CLI Mode Parser. Input Arguments are 'file_path': Path to input file/folder and 'filetype': Type of input (file/folder)"""
    print("Running as CLI Tool!")
    session = start_inference_session()
    results = []
    if filepath.suffix.lower() in VALID_EXTENSIONS or filetype=="folder":  
        target_data = img_search(filepath)
        for image in target_data:
            results.append(infer_one(str(image),session)) # Cast WindowsPath object into string
    elif filepath.suffix.lower()==".zip": # zips need their own handler
        target_data = zip_parser(filepath)
        for dict_object in target_data:
            results.append(infer_one(dict_object["content"],session))
    print_results(results)