import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from backend.recomendations import disease_info
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "finetuned_plant_disease_detection_model.keras"), compile=False)
print("Model loaded successfully")
with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)
print("Class names loaded successfully")
def predict_disease(img_path):
    input_image=image.load_img(img_path,target_size=(224,224))
    np_image=image.img_to_array(input_image)
    np_image=np_image/255.0
    np_image=np.expand_dims(np_image,axis=0)
    prediction=model.predict(np_image)
    argmax=np.argmax(prediction)
    confidence=np.max(prediction)*100
    predicted_class=class_names[argmax]
    return predicted_class,confidence

# Example usage
while True:
    print("Enter 'exit' to quit the program.")
    img_path=input("Enter the path to the plant image: ")
    if img_path == "exit":
        break
    if not os.path.exists(img_path):
        print("Image path does not exist. Please provide a valid path.")
    else:
        predicted_class, confidence = predict_disease(img_path)
        print(f"Predicted disease: \n{predicted_class}\n")
        print(f"Confidence: \n{confidence:.2f}%\n")

        if predicted_class in disease_info:
            info=disease_info[predicted_class]
            print(f"Cause: \n{info['cause']}")
            print()
            print(f"Treatment: \n{info['treatment']}")
            print()
            print(f"Prevention: \n{info['prevention']}")  
        else:
            print("No additional information available for this disease.")

