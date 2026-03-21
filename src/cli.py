import argparse
from src.scripts.inference import infer_one
from src.utils.util_functions import load_saved_model

model = load_saved_model()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion-MNIST inference")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )
    args = parser.parse_args()

    results = infer_one(args.image,model)
    print(f"Class: {results["predicted_class"]}")
    print(f"Confidence: {results["confidence"]}")