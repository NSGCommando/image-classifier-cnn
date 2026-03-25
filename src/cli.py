from src.scripts.inference import infer_one
from src.utils.util_functions import load_saved_model
# Run this for the CLI mode
def cli_inference(image_data):
    print("Running as CLI Tool!")
    model = load_saved_model()
    results = infer_one(image_data,model)
    print(f"Class: {results["predicted_class"]}")
    print(f"Confidence: {results["confidence"]}")