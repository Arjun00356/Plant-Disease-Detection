from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("models/plant_model.h5")

class_names = list(...)   # load from training generator

def preprocess(img):
    img = img.resize((224,224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_arr = preprocess(image)

    pred = model.predict(input_arr)
    idx = np.argmax(pred)
    label = class_names[idx]

    status = "Healthy" if "healthy" in label.lower() else "Diseased"

    return {
        "plant": label.split("___")[0],
        "condition": label.split("___")[1],
        "status": status,
        "confidence": float(np.max(pred))*100
    }
