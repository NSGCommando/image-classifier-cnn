from src.scripts.inference import infer_one
from src.utils.util_functions import img_search, load_saved_model, print_results
# Run this for the CLI mode
def cli_inference(image_path):
    print("Running as CLI Tool!")
    target_data = img_search(image_path)
    model = load_saved_model()
    results = []
    for image in target_data:
        results.append(infer_one(str(image),model)) # Cast WindowsPath object into string
    print_results(results)