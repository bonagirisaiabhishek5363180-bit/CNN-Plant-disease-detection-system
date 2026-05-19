import json
import os

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tensorflow.keras.models import load_model

from .recommendations import disease_info


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")


with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model(
    os.path.join(BASE_DIR, "finetuned_plant_disease_detection_model.keras")
)

plant_model = load_model(
    os.path.join(BASE_DIR, "plant_nonplant_classifier.keras")
)


plant_class_names = ["leaf", "non-leaf"]


def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image


def predict_plant_nonplant(image):

    img = preprocess_image(image)

    prediction = plant_model.predict(img)

    argmax = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_class = plant_class_names[argmax]

    return predicted_class, confidence


def predict_disease(image):

    img = preprocess_image(image)

    prediction = model.predict(img)

    argmax = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_class = class_names[argmax]

    if predicted_class in disease_info:
        info = disease_info[predicted_class]
    else:
        info = None

    return predicted_class, confidence, info


app.mount(
    "/static",
    StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=os.path.join(FRONTEND_DIR, "templates")
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file)

    plant_result, plant_confidence = predict_plant_nonplant(image)

    if plant_result == "non-leaf" or plant_confidence < 70:

        return JSONResponse(
            {
                "disease": "Invalid or non-plant image",
                "confidence": round(float(plant_confidence), 2),
                "cause": None,
                "prevention": None,
                "treatment": None,
            }
        )

    predicted_class, confidence, info = predict_disease(image)

    return JSONResponse(
        {
            "disease": predicted_class,
            "confidence": round(float(confidence), 2),
            "cause": info["cause"] if info else None,
            "prevention": info["prevention"] if info else None,
            "treatment": info["treatment"] if info else None,
        }
    )