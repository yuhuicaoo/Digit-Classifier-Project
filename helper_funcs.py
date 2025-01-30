import torch
from model import ANN
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_digit(model, img_tensor):
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = F.softmax(logits, dim=1)
        _ , preds = torch.max(probabilities, 1)
        probabilities = [round(prob * 100, 2) for prob in probabilities.squeeze().tolist()]
        return preds.item(), probabilities
    
def preprocess_img(image):
    transforms = v2.Compose([
        v2.Resize((28,28)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    img_tensor = transforms(image) 

    img_array = img_tensor.squeeze().numpy()
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')
    plt.show()

    return img_tensor.unsqueeze(0)