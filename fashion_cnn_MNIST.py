import keras as ks
import tensorflow as tf
from fashion_cnn_functions import new_model, preprocess_dataset

# load stored evaluation accuracy
prev_eval_accuracy = 0
with open("eval_accuracy.txt","r") as f:
    prev_eval_accuracy = float(f.read())

raw_set = ks.datasets.fashion_mnist
train_set, test_set = preprocess_dataset(raw_set)

# train model and save weights
model = new_model()
history = model.fit(train_set,epochs=10, validation_data=test_set)
loss, acc = model.evaluate(test_set)

# save weights if eval accuracy is better than before
if acc>prev_eval_accuracy:
    model.save_weights("fashion_cnn.weights.h5")
    with open("eval_accuracy.txt","w") as f:
        f.write(f"{acc:.4f}")