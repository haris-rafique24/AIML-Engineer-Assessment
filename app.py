#run using uvicorn app:app --reload
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
from torchvision import transforms
from model_factory import get_model
import config

app = FastAPI()

# --- Explicit model loading  ---
model = get_model(num_classes=config.NUM_CLASSES)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Preprocessing code block
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    # 2. Inference
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # 3. Format output with basic thresholding
    res = prediction[0]
    thresh = 0.5
    mask = res['scores'] > thresh
    
    return {
        "boxes": res['boxes'][mask].tolist(), 
        "labels": res['labels'][mask].tolist(),
        "scores": res['scores'][mask].tolist()
    }