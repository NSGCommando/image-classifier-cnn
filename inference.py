import argparse
import keras.ops as ops
from utils.fashion_cnn_functions import new_model, load_and_preprocess_image, Paths
modelpath = Paths.ModelsPath.value/"fashion_cnn.weights.keras"
# names of classes from MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def main(image_path):
    model = new_model()
    model.load_weights(modelpath)

    img = load_and_preprocess_image(image_path)
    predictions = model.predict(x=img)
    pred_tensor = ops.argmax(predictions, axis=1)[0]
    conf_tensor = ops.max(predictions)

    predicted_class = int(ops.convert_to_numpy(pred_tensor))#type: ignore
    confidence = float(ops.convert_to_numpy(conf_tensor))#type: ignore

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