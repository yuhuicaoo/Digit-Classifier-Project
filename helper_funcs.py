import torch
from model import ANN
from torchvision.transforms import v2
import matplotlib.pyplot as plt

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_digit(model, img_tensor):
    with torch.no_grad():
        out = model(img_tensor)
        _, preds = torch.max(out.data,1)
        return preds.item()
    
def preprocess_img(image):
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((28,28)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    img_tensor = transforms(image) 

    img_array = img_tensor.squeeze().numpy()
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')
    plt.show()

    img_tensor = (img_tensor > 0).float()
    print(img_tensor)

    return img_tensor.unsqueeze(0)