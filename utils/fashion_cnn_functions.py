import keras as ks
import keras.ops as ops
from enum import Enum
from keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path
class Paths(Enum):
    ModelsPath = Path(__file__).parents[1].resolve()/"models"
    ResultsPath = Path(__file__).parents[1].resolve()/"results"

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
        ks.layers.Dense(10)
    ])
    # Compile the model with optimizer and loss function using accuracy as the measurement
    opt = ks.optimizers.Adam(0.001)
    the_model.compile(optimizer=opt,#type: ignore
                      loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
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

# def preprocess_dataset(d1):
#     (xtrain, ytrain), (xtest, ytest) = d1.load_data()

#     # Scale image pixel values to between 0 and 1
#     train_images = xtrain[..., tf.newaxis] / 255.0
#     test_images = xtest[..., tf.newaxis] / 255.0

#     # Create and filter the datasets for desired classes
#     train_set = tf.data.Dataset.from_tensor_slices((train_images, ytrain)).batch(32)
#     test_set = tf.data.Dataset.from_tensor_slices((test_images, ytest)).batch(32)
#     return train_set, test_set

def preprocess_dataset(train_data, test_data):
    """
    Helper to normalize and batch training and testing dat.
    Returns two batched Datasets.
    """
    (xtrain, ytrain) = train_data
    (xtest, ytest) = test_data

    # Scale using Keras ops (replaces tf.newaxis and manual division)
    train_images = ops.expand_dims(xtrain, axis=-1)
    test_images = ops.expand_dims(xtest, axis=-1)
    # Need explicit casting to Float32 from Uint8
    train_images = ops.cast(train_images, dtype="float32") / 255.0
    test_images = ops.cast(test_images, dtype="float32") / 255.0

    return (train_images, ytrain), (test_images, ytest)

def load_and_preprocess_image(img_path):
    """Helper to load and preprocess a single iamge during inference run.
    Returns an image array"""
    img = image.load_img(
        img_path,
        color_mode="grayscale",
        target_size=(28, 28)
    )
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = ops.expand_dims(img_array, axis=0)
    return img_array