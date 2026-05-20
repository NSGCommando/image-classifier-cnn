from functools import wraps
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import List
from fastapi import HTTPException, UploadFile
from zipfile import ZipFile
from uuid import uuid4
from pathlib import Path
from onnxruntime import InferenceSession
from src.utils.constants import Paths, ImageData, VALID_EXTENSIONS
import subprocess

def start_inference_session()->InferenceSession:
    return InferenceSession(Paths.ModelsPath.value/"onnx_model.onnx")

def default_parser():
    """
    Helper to create a parser with default argument flags.
    Returns an argument Namespace object.
    """
    parser = ArgumentParser(
        description="Fashion-MNIST inference",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        Usage:
        CLI File Inference:  python run.py --file [path To image/Zip]\n
        CLI Local Directory Inference: python run.py --folder [path To folder]\n
        Web Server:     python run.py --server
        """
        )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--file",
        type=str,
        help="Path to input image or zipfile"
    )
    modes.add_argument(
        "--folder",
        type=str,
        help="Path to local folder containing images"
    )
    modes.add_argument(
        "--server",
        action="store_true",
        help="Option to start FastAPI Uvicorn server"
    )
    return parser.parse_args()

# Finds all valid images within provided path
def img_search(dir_path:Path):
    """
    Simple helper for retrieving paths of all valid images inside provided directory.
    Also searchs within sub-directories via 'rglob'.
    Yields an generator for providing absolute image filepath objects (pathlib.WindowsPath)
    """
    if dir_path.is_file():
        yield dir_path.resolve()
    else:
        all_img_path_generator = dir_path.rglob("*") # rglob is Recursive Glob, searches sub-directories
        # use a Set to check for Membership of the filetype. Also resolve all valid image paths
        image_path_generator = (img_path.resolve() for img_path in all_img_path_generator if img_path.suffix.lower() in VALID_EXTENSIONS)
        yield from image_path_generator

# print results for CLI Mode
def print_results(result_data):
    """Helper to print CLI mode inference results to console"""
    for img in result_data:
        print(f"Class: {img["predicted_class"]}")
        print(f"Confidence: {img["confidence"]}")


def zip_parser(zip_path)->List[ImageData]:
    """Zip parser for zip files. Takes path to the zipfile. 
    Returns a List of Dicts of format {'filename':string,'content':bytes}."""
    image_data_list=[]
    with ZipFile(zip_path, "r") as z:
        for file_info in z.infolist():
            if file_info.is_dir():continue # skip sub-directories
            if Path(file_info.filename).suffix.lower() in VALID_EXTENSIONS: # only open image files
                with z.open(file_info) as image_file:
                    content = image_file.read()
                    # append raw bytes for each image, handled by caller
                    image_data_list.append({
                        "filename": file_info.filename,
                        "content": content
                    })
    return image_data_list

# decorator for boilerplate HTTP filetype check and handling zips
def image_handler_http(f):
    """
    Decorator to check filetype and handle if it is a zip.
    Returns a dictionary of [filename,content], where filename is the image filename and content is the image's raw bytes
    """
    @wraps(f)
    async def edited_f(*args,**kwargs):
        image_data_list = []
        target_file:UploadFile|None = kwargs.get("file") # The UploadFile defined in the API route is passed as a keyword arg to decorator
        if not target_file: raise HTTPException(status_code=400,detail="Upload failed/file not found")
        if not target_file.filename:raise HTTPException(status_code=400,detail="Uploaded File has no Filename")
        if not target_file.content_type:raise HTTPException(status_code=400,detail="Uploaded File has incorrect headers")
        if not target_file.content_type.startswith(("image/","application/zip")):
            raise HTTPException(status_code=400,detail="Uploaded File is not a Zip or Image")
        if target_file.content_type.startswith("image"): # check if single image first
            content = await target_file.read() # need to await UploadFile
            image_data_list.append({
                                "filename": target_file.filename,
                                "content": content
                            })
        # handle zip upload
        elif target_file.content_type.startswith("application/zip"):
            unique_id = uuid4().hex
            temp_path = f"src/results/tempzip_{unique_id}.zip"
            try:
                with open(temp_path, "wb") as buffer:
                    # reads the file in chunks and saves to temp location
                    while chunk := await target_file.read(1024 * 64):
                        buffer.write(chunk)
                image_data_list = zip_parser(temp_path)
            finally: # delete the temp zip file after we're done
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
        # inject the image data
        zipname = target_file.filename
        return await f(image_data_list=image_data_list, zipname=zipname) # need await if the wrapped func is async
    return edited_f

# server subprocess runner
def run_server(command, proc_name):
    """
    Runs a server as a subprocess and prints its output to parent process.
    Graceful shutdown via keyboard interrupts handled.
    """
    process = subprocess.Popen(
            args=command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    try:
        for line in process.stdout:#type:ignore
            print(f"[{proc_name}] {line.strip()}")
    
    except KeyboardInterrupt:
        print(f"[{proc_name}] KeyboardInterrupt caught, terminating server...")
        process.terminate()
        process.wait()
    finally:
        if process.poll() is None: # check if child process has terminated
            process.terminate()
            process.wait()
        print(f"[{proc_name}] server exited cleanly.")

    process.wait()
    print(f"[{proc_name}] exited with code {process.returncode}")