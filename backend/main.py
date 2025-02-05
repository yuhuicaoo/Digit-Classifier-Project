from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from model import CNN
from utils import load_model, predict_digit
from io import BytesIO
import torch
from torchvision.transforms import v2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CNN(num_class=10)
model = load_model(model, "model/cnn_model.pth")

def preprocess_image(contents: bytes):
    image = Image.open(BytesIO(contents)).convert("L")
    transforms = v2.Compose([
        v2.Resize((28,28)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    img_tensor = transforms(image)
    return img_tensor.unsqueeze(0)

@app.post("/")
async def prediction(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img_tensor = preprocess_image(data)
        prediction, probabilities = predict_digit(model, img_tensor)

        return {
            "digit": prediction,
            "probabilities": probabilities
        }
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}

