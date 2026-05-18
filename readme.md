# 🌿 AI-Powered Plant Disease Detection System

## 🌐 Live Demo

[https://cnn-plant-disease-detection-syste](https://cnn-plant-disease-detection-syste)

An end-to-end Deep Learning web application that detects plant diseases from leaf images using Transfer Learning with MobileNetV2.

The system predicts diseases from uploaded plant leaf images and provides:

* Disease prediction
* Confidence score
* Cause of disease
* Treatment recommendations
* Prevention methods

---

# 🚀 Features

✅ Deep Learning-based plant disease classification
✅ Transfer Learning using MobileNetV2
✅ FastAPI backend for inference
✅ Interactive frontend for image upload
✅ Recommendation system for treatments and prevention
✅ Fine-tuned TensorFlow model
✅ Training & validation accuracy visualization
✅ Real-time image prediction
✅ Deployment-ready architecture

---

# 🧠 Model Architecture

The project uses:

* TensorFlow
* Keras
* MobileNetV2 (Pretrained on ImageNet)
* Transfer Learning
* Fine-Tuning

### Training Techniques Used

* Data Augmentation
* Transfer Learning
* Fine-Tuning last MobileNetV2 layers
* Early Stopping
* Validation Split
* Learning Rate Optimization

---

# 📊 Model Performance

| Metric              | Value          |
| ------------------- | -------------- |
| Training Accuracy   | ~95%           |
| Validation Accuracy | ~96%           |
| Validation Loss     | ~0.11          |
| Classes             | 15             |
| Dataset Size        | 20,000+ Images |

---

# 📈 Accuracy Graph

<img src="accuracy_plot.png" width="700">

---

# 📉 Loss Graph

<img src="loss_plot.png" width="700">

---

# 🖼️ Sample Input Image

<img src="sample_input.png" width="500">

---

# ✅ Sample Prediction Output

<img src="sample_output.png" width="700">

---

# 🧪 Supported Plant Diseases

### Pepper

* Pepper__bell___Bacterial_spot
* Pepper__bell___healthy

### Potato

* Potato___Early_blight
* Potato___Late_blight
* Potato___healthy

### Tomato

* Tomato_Bacterial_spot
* Tomato_Early_blight
* Tomato_Late_blight
* Tomato_Leaf_Mold
* Tomato_Septoria_leaf_spot
* Tomato_Spider_mites_Two_spotted_spider_mite
* Tomato__Target_Spot
* Tomato__Tomato_YellowLeaf__Curl_Virus
* Tomato__Tomato_mosaic_virus
* Tomato_healthy

---

# 🏗️ Tech Stack

## Machine Learning

* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Matplotlib

## Backend

* FastAPI
* Uvicorn
* Python

## Frontend

* HTML
* CSS
* JavaScript

---

# 📂 Project Structure

```text
CNN-Plant disease detection system/
│
├── backend/
│   ├── main.py
│   ├── predict.py
│   ├── recomendations.py
│   ├── class_names.json
│   └── finetuned_plant_disease_detection_model.keras
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── accuracy_plot.png
├── loss_plot.png
├── sample_input.png
├── sample_output.png
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <your-github-repo-link>
cd CNN-Plant-disease-detection-system
```

---

# 📦 Create Virtual Environment

```bash
python -m venv .venv
```

---

# ▶️ Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

# 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run Backend

```bash
uvicorn main:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

# 🌐 Open Frontend

Open:

```text
frontend/index.html
```

in browser.

---

# 🔄 Workflow

```text
User uploads leaf image
            ↓
Frontend sends image to FastAPI
            ↓
TensorFlow model performs inference
            ↓
Prediction + confidence generated
            ↓
Recommendation system returns treatment details
            ↓
Frontend displays results
```

---

# ☁️ Deployment Ready

This project is designed for deployment using:

* Render

---

# 📌 Future Improvements

* Real-time camera prediction
* Multi-language support
* Disease severity estimation
* Farmer dashboard
* Cloud database integration
* Mobile app deployment
* TensorFlow Lite optimization

---

# 👨‍💻 Developer

Sai Abhishek

Second-Year Engineering Student passionate about:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* AI Deployment


---

# ⭐ If You Like This Project

Give this repository a star on GitHub.
