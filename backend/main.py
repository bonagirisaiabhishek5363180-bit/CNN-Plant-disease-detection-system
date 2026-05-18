
import json
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tensorflow.keras.models import load_model

from .recomendations import disease_info

app = FastAPI()

model = load_model("backend/finetuned_plant_disease_detection_model.keras")

with open("backend/class_names.json", "r") as f:
    class_names = json.load(f)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


def predict_disease(image):

    image = image.resize((224, 224))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    argmax = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_class = class_names[argmax]

    info = disease_info[predicted_class]

    return predicted_class, confidence, info


@app.get("/")
def home():

    return FileResponse("frontend/index.html")


@app.post("/")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file)

    predicted_class, confidence, info = predict_disease(image)

    return {
        "disease": predicted_class,
        "confidence": round(float(confidence), 2),
        "cause": info["cause"],
        "treatment": info["treatment"],
        "prevention": info["prevention"]
    }