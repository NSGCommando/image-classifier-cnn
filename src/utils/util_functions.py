import keras as ks
from keras.ops import cast, expand_dims
from enum import Enum
from argparse import ArgumentParser
from keras.models import load_model, Model
import matplotlib.pyplot as plt
from pathlib import Path

class Paths(Enum):
    ModelsPath = Path(__file__).parents[2].resolve()/"models"
    ResultsPath = Path(__file__).parents[2].resolve()/"results"

def new_model()->ks.Sequential:
    """Factory for new model instance.
    Returns a new instance of Sequential model"""
    the_model = ks.models.Sequential([
        ks.layers.Input(shape=(28,28,1)),
        ks.layers.Conv2D(32, (3, 3)),
        ks.layers.BatchNormalization(),
        ks.layers.Activation(ks.activations.relu),
        ks.layers.MaxPooling2D((2, 2), strides=2),

        ks.layers.Conv2D(32, (3, 3),padding='same'),
        ks.layers.BatchNormalization(),
        ks.layers.Activation(ks.activations.relu),
        ks.layers.MaxPooling2D((2, 2), strides=2),

        ks.layers.Flatten(),
        ks.layers.Dense(128, activation='relu'),
        ks.layers.Dense(256, activation='relu',kernel_regularizer=ks.regularizers.l2(1e-4)),
        ks.layers.Dropout(0.2),
        ks.layers.Dense(10, activation='softmax')
    ])
    # Compile the model with optimizer and loss function using accuracy as the measurement
    opt = ks.optimizers.Adam(0.001)
    the_model.compile(optimizer=opt,#type: ignore
                      loss=ks.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[ks.metrics.SparseCategoricalAccuracy()])
    return the_model


# Functions to show sample of dataset image
def show_example_color(images):
    """Helper"""
    plt.figure()
    plt.imshow(images)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# Function to check if labels are correct after filtering
def print_label_example(d1):
    """Helepr to print a label alongside the corresponding image."""
    for image, label in d1:
        print("Label:", label.numpy())
        show_example_color(image)
        break

def preprocess_dataset(train_data, test_data):
    """
    Helper to normalize and batch training and testing dat.
    Returns two batched Datasets.
    """
    (xtrain, ytrain) = train_data
    (xtest, ytest) = test_data

    # Scale using Keras ops (replaces tf.newaxis and manual division)
    train_images = expand_dims(xtrain, axis=-1)
    test_images = expand_dims(xtest, axis=-1)
    # Need explicit casting to Float32 from Uint8
    train_images = cast(train_images, dtype="float32") / 255.0
    test_images = cast(test_images, dtype="float32") / 255.0

    return (train_images, ytrain), (test_images, ytest)

def load_saved_model(modelpath=Paths.ModelsPath.value/"fashion_cnn.weights.keras"):
    """Helper to load saved model from specified path.
    Returns an instance of keras.models.Model"""
    model = load_model(modelpath)
    if not isinstance(model,Model):
        raise RuntimeError("Model didn't load!")
    return model

def default_parser():
    """
    Helper to create a parser with default argument flags.
    Returns an argument Namespace object.
    """
    parser = ArgumentParser(
        description="Fashion-MNIST inference",
        epilog="""
        Usage:
        1. CLI Inference:  python run.py --image path/to/img.jpg
        2. Web Server:     python run.py
        """
        )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )
    modes.add_argument(
        "--server",
        action="store_true",
        help="Option to start FastAPI Uvicorn server"
    )
    return parser.parse_args()