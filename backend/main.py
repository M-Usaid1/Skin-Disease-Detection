from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import shutil
import os

app = FastAPI(
    title="Skin Disease Detection API",
    description="YOLOv11 Acne Detection",
    version="1.0.0"
)

# -----------------------------
# Create required directories
# -----------------------------
UPLOAD_DIR = "uploads"
PREDICTION_DIR = "predictions"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

app.mount(
    "/predictions",
    StaticFiles(directory=PREDICTION_DIR),
    name="predictions"
)

# -----------------------------
# Load trained model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Skin Disease Detection API is Running"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded image
    image_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    results = model.predict(
        source=image_path,
        save=True,
        conf=0.10,
        project="predictions",
        name="result",
        exist_ok=True
    )
    print(results)
    print("Boxes:", results[0].boxes)

    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class": model.names[cls],
                "confidence": round(conf, 3)
            })

    prediction_image = os.path.join(
        "predictions",
        "result",
        os.path.basename(image_path)
    )

    return JSONResponse({
        "message": "Prediction Successful",
        "detections": detections,
        "prediction_image": prediction_image
    })