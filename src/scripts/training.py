import keras as ks
from src.utils.util_functions import new_model, preprocess_dataset, Paths

modelsdir = Paths.ModelsPath.value
modelsdir.mkdir(parents=True, exist_ok=True)
evaldir = Paths.ResultsPath.value

# load stored evaluation accuracy
accuracy_store = "eval_accuracy.txt"
prev_eval_accuracy = 0
with open(evaldir/accuracy_store,"r") as f:
    data = f.read().strip()
    prev_eval_accuracy = float(data)

raw_data = ks.datasets.fashion_mnist.load_data()
(train_images,train_labels), (test_images,test_labels) = preprocess_dataset(raw_data[0], raw_data[1])

# train model and save weights
model = new_model()
history = model.fit(x=train_images,y=train_labels,epochs=10, validation_data=(test_images,test_labels))
loss, acc = model.evaluate(x=test_images,y=test_labels)

# save weights if eval accuracy is better than before
if acc>prev_eval_accuracy:
    model.save(filepath=modelsdir/"fashion_cnn.weights.keras")
    with open(file=evaldir/accuracy_store,mode="w") as f:
        f.write(f"{acc:.4f}")