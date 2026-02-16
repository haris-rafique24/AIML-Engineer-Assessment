from fastapi import FastAPI, File, UploadFile, Response
import torch
from PIL import Image, ImageDraw, ImageFont
import io
from torchvision import transforms
from model_factory import get_model
import config

app = FastAPI()

# Map label IDs to names from the BCCD dataset
ID_TO_LABEL = {1: "WBC", 2: "RBC", 3: "Platelets"}

model = get_model(num_classes=config.NUM_CLASSES)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
model.eval()

@app.post("/predict_visual")
async def predict_visual(file: UploadFile = File(...)):
    # 1. Preprocessing
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    # 2. Inference
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # 3. Drawing Boxes, Labels, and Scores
    draw = ImageDraw.Draw(image)
    res = prediction[0]
    
    for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
        if score > 0.5: # Thresholding
            box_list = box.tolist()
            label_name = ID_TO_LABEL.get(label.item(), "Unknown")
            display_text = f"{label_name}: {score:.2f}"
            
            # Draw Rectangle
            draw.rectangle(box_list, outline="red", width=3)
            
            # Draw Label Background and Text
            # Note: For better visibility, we draw a small filled box behind the text
            text_pos = (box_list[0], box_list[1] - 15)
            draw.text(text_pos, display_text, fill="green")
    
    # 4. Return processed image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
