# Image Classifier #
## Architecture ##
- CNN-based image classifier trained on MNIST dataset
- Structure:
 - Input layer with input shape of 28x28 pixels per image
 - Two blocks of Conv2d layers, blocks structured as follows: ```COnv2d->BatchNorm->ReLUActivation->MaxPooling```
 - Conv2d blocks followed by two dense layers of 128 and 256 neurons each, second dense layer has L2 Regularization applied to it
 - Final layer is dense layer for classification with 10 output neurons

## Implemented ##
- Training
- Weight Loading and Saving
- Single-Image Inference

## Inference ##
- To run inference on a single image, use the following command in terminal from the root folder: ```python inference.py --image [image_path]```
- If running in a virtual environment, and Tensorflow is not installed globally, venv scripts should be enabled first.
  - If terminal is Windows PowerShell, it might be necessary to bypass PowerShell's script execution restrictions for current terminal session using ```Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass```
  - Followed by ```.\venv\Scripts\Activate.ps1``` to activate scripts then running inference command above
