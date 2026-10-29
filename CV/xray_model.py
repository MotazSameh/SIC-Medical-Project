"""Chest X-ray (NORMAL vs PNEUMONIA) inference.
DenseNet121 exported from CV_mlflow_v2.ipynb (inference only).
"""
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import zipfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

SRC_PATH = Path(__file__).parent / "densenet121_threshold_046.keras"
MODEL_PATH = Path(__file__).parent / "densenet121_threshold_046_k312.keras"
IMG_SIZE = (224, 224)


@keras.saving.register_keras_serializable(package="Custom", name="FixedThreshold")
class FixedThreshold(keras.layers.Layer):
    def __init__(self, threshold=0.37, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs):
        return tf.cast(inputs >= self.threshold, tf.float32)

    def get_config(self):
        config = super().get_config()
        config["threshold"] = self.threshold
        return config


def _strip_key(obj, key):
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _strip_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            _strip_key(v, key)


def _ensure_compatible_model():
    
    if MODEL_PATH.exists():
        return
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {SRC_PATH}")
    with zipfile.ZipFile(SRC_PATH) as zin, \
            zipfile.ZipFile(MODEL_PATH, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data)
                _strip_key(cfg, "quantization_config")
                data = json.dumps(cfg).encode("utf-8")
            zout.writestr(item, data)

_model = None


def get_model():
    global _model
    if _model is None:
        _ensure_compatible_model()
        _model = keras.saving.load_model(
            MODEL_PATH,
            custom_objects={"FixedThreshold": FixedThreshold},
            compile=False,
        )
    return _model


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)  
    arr = keras.applications.densenet.preprocess_input(img.numpy())
    return arr[None, ...].astype("float32")


def predict(image_bytes: bytes) -> dict:
    model = get_model()
    out = model(preprocess(image_bytes), training=False)
    prob = float(np.asarray(out["pneumonia_probability"]).ravel()[0])
    is_pneumonia = int(np.asarray(out["prediction"]).ravel()[0]) == 1
    threshold = float(model.get_layer("fixed_threshold").threshold)
    return {
        "label": "PNEUMONIA" if is_pneumonia else "NORMAL",
        "pneumonia_probability": round(prob, 4),
        "threshold": round(threshold, 4),
    }


if __name__ == "__main__":
    import sys

    m = get_model()
    print("Inputs :", m.input_shape)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            print(predict(f.read()))