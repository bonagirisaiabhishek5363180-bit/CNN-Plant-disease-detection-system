import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from recomendations import disease_info

with open("class_names.json", "rb") as f:
    class_names = json.load(f)

app = FastAPI()
model = load_model("finetuned_plant_disease_detection_model.keras")


def predict_disease(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    argmax = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_names[argmax]
    if predicted_class in disease_info:
        info = disease_info[predicted_class]
    else:
        info = None
    return predicted_class, confidence, info


app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
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
    image = Image.open(file.file)
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
