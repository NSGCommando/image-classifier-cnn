```Image Classifier```
- CNN-based image classifier trained on MNIST dataset

```Implemented```
- Training
- Weight Loading and Saving
- Single-Image Inference

```Inference```
- To run inference on a single image, use the following command in terminal from the root folder: ```python inference.py --image [image_path]```
- If running in a virtual environment, and Tensorflow is not installed globally, venv scripts should be enabled first.
  - If terminal is Windows PowerShell, it might be necessary to bypass PowerShell's script execution restrictions for current terminal session using ```Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass```
  - Followed by ```.\venv\Scripts\Activate.ps1``` to activate scripts then running inference command above
