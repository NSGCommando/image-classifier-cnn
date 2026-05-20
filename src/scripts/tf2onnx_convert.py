"""
Helper script to convert 'saved_model.pb' into 'onnx_model.onnx'.
Run via 'python -m src.scripts.tf2onnx_convert' from project root.
"""
from pathlib import Path
import subprocess, sys
from src.utils.constants import Paths
saved_model_path = Paths.ModelsPath.value/"Exported_Model"
output_path = Paths.ModelsPath.value / "onnx_model.onnx"
project_root = Paths.RootPath.value
print(project_root)

# Use 'sys.executable' to ensure the project's .venv is activated for access to tf2onnx
conversion_cmd = [sys.executable,"-m","tf2onnx.convert", "--saved-model", str(saved_model_path), "--output", str(output_path)]
subprocess.run(args=conversion_cmd, cwd=project_root)