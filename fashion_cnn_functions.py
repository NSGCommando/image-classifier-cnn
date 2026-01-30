import keras as ks
import tensorflow as tf
import matplotlib.pyplot as plt

def new_model():
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
    the_model.compile(optimizer=ks.optimizers.Adam(0.001),
                      loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[ks.metrics.SparseCategoricalAccuracy()])
    return the_model


# Functions to show sample of dataset image
def show_example_color(images):
    plt.figure()
    plt.imshow(images)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# Function to check if labels are correct after filtering
def print_label_example(d1):
    for image, label in d1:
        print("Label:", label.numpy())
        show_example_color(image)
        break

def preprocess_dataset(d1):
    (xtrain, ytrain), (xtest, ytest) = d1.load_data()

    # Scale image pixel values to between 0 and 1
    train_images = xtrain[..., tf.newaxis] / 255.0
    test_images = xtest[..., tf.newaxis] / 255.0

    # Create and filter the datasets for desired classes
    train_set = tf.data.Dataset.from_tensor_slices((train_images, ytrain)).batch(32)
    test_set = tf.data.Dataset.from_tensor_slices((test_images, ytest)).batch(32)
    return train_set, test_set