import argparse
import numpy as np
from tensorflow.keras.preprocessing import image
from fashion_cnn_functions import new_model

# names of classes from MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
def load_and_preprocess_image(img_path):
    img = image.load_img(
        img_path,
        color_mode="grayscale",
        target_size=(28, 28)
    )
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main(image_path):
    model = new_model()
    model.load_weights("fashion_cnn.weights.h5")

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