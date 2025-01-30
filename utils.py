import torch
from model import ANN
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math

def visualise_samples(train_loader, num_samples=10):

    current_image_batch, current_image_label = next(iter(train_loader))
    display_rows = math.ceil(num_samples / 5)

    plt.figure(figsize=(8,6))
    for i in range(min(num_samples, current_image_batch.size(0))):
        plt.subplot(display_rows, 5, i + 1)
        plt.imshow(current_image_batch[i].squeeze(), cmap='gray')
        plt.title(f"Label : {current_image_label[i].item()}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    plt.figure(figsize=(8,6))

    # Plot losses
    plt.subplot(1,2,1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()

    # Plot Accuracies
    plt.subplot(1,2,2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

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

    # img_array = img_tensor.squeeze().numpy()
    # plt.imshow(img_array, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return img_tensor.unsqueeze(0)