import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from backend.recommendations import disease_info
from fastapi.middleware.cors import CORSMiddleware

with open("backend/class_names.json", "r") as f:
    class_names = json.load(f)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("backend/finetuned_plant_disease_detection_model.keras")
plant_model = load_model("backend/plant_nonplant_classifier.keras")
plant_class_names = ["leaf", "non-leaf"]


def normalize_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def predict_plant_nonplant(image):
    img = normalize_image(image)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = plant_model.predict(img)
    argmax = np.argmax(prediction)
    predicted_class = plant_class_names[argmax]
    return predicted_class


def predict_disease(image):
    img = normalize_image(image)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    argmax = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_names[argmax]
    info = disease_info.get(predicted_class)
    return predicted_class, confidence, info


app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static",
)
templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request}
    )


@app.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
    except Exception:
        return JSONResponse(
            {
                "error": "Invalid image file",
                "disease": "Invalid image",
                "confidence": 0.0,
                "cause": None,
                "prevention": None,
                "treatment": None,
            },
            status_code=400,
        )

    try:
        plant_nonplant_result = predict_plant_nonplant(image)
    except Exception:
        return JSONResponse(
            {
                "error": "Could not classify plant vs non-plant",
                "disease": "Error processing image",
                "confidence": 0.0,
                "cause": None,
                "prevention": None,
                "treatment": None,
            },
            status_code=500,
        )

    if plant_nonplant_result == "non-leaf":
        return JSONResponse(
            {
                "disease": "Not a plant",
                "confidence": 0.0,
                "cause": None,
                "prevention": None,
                "treatment": None,
            }
        )

    try:
        predicted_class, confidence, info = predict_disease(image)
    except Exception:
        return JSONResponse(
            {
                "error": "Failed to predict disease",
                "disease": "Error processing image",
                "confidence": 0.0,
                "cause": None,
                "prevention": None,
                "treatment": None,
            },
            status_code=500,
        )

    if confidence < 45:
        return JSONResponse(
            {
                "disease": "Low confidence in prediction",
                "confidence": round(float(confidence), 2),
                "cause": None,
                "prevention": None,
                "treatment": None,
            }
        )

    return JSONResponse(
        {
            "disease": predicted_class,
            "confidence": round(float(confidence), 2),
            "cause": info["cause"] if info else None,
            "prevention": info["prevention"] if info else None,
            "treatment": info["treatment"] if info else None,
        }
    )
