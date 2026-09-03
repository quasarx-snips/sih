import os
import cv2
import numpy as np
import onnxruntime as ort

IMAGE_PATH = os.path.join("inputs", "sample_surface.jpg") # Replace with your actual image filename
MODEL_PATH = os.path.join("models", "crater_unet.onnx")

print("[1/4] Checking file paths...", flush=True)
assert os.path.exists(IMAGE_PATH), f"Missing image at: {os.path.abspath(IMAGE_PATH)}"
assert os.path.exists(MODEL_PATH), f"Missing model at: {os.path.abspath(MODEL_PATH)}"

print("[2/4] Loading ONNX Session on CPU...", flush=True)
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

print("[3/4] Reading image...", flush=True)
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Failed to read image with OpenCV!"
print(f"Image loaded: {img.shape}", flush=True)

print("[4/4] Running single tile inference test...", flush=True)
tile = cv2.resize(img, (512, 512))
inp = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)

out = session.run(None, {session.get_inputs()[0].name: inp})
print(f"SUCCESS! Inference complete. Output shape: {out[0].shape}", flush=True)