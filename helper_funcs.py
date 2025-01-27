import torch
from model import ANN
from torchvision.transforms import v2

def load_model(model_path):
    model = ANN(28 * 28, 10)
    model.load_state_dict(torch.load("model/neural_network_model.pth", weights_only=True))
    model.eval()
    return model

def predict_digit(model, img_tensor):
    with torch.no_grad():
        out = model(img_tensor)
        _, preds = torch.max(out.data,1)
        return preds.item()
    
def preprocess_img(image):
    transforms = v2.Compose([
        v2.Resize((28,28)),
        v2.PILToTensor()
    ])

    img_tensor = transforms(image).float() / 255.0

    return img_tensor.unsqueeze(0)