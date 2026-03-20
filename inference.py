import argparse
import numpy as np
from utils.fashion_cnn_functions import new_model, load_and_preprocess_image
# names of classes from MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def main(image_path):
    model = new_model()
    model.load_weights("fashion_cnn.weights.keras")

    img = load_and_preprocess_image(image_path)
    predictions = model.predict(img)

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    print(f"Predicted class: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion-MNIST inference")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )
    args = parser.parse_args()

    main(args.image)