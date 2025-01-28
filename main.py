import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math
from model import ANN, CNN
from train import train_model, evaluate_model
from get_data import load_data

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

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


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 100
    input_size = 28 * 28
    num_class = 10
    num_epochs = 10
    learning_rate = 0.001

    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size)

    # Visualise a sample of images from training dataset
    visualise_samples(train_loader, num_samples=10)

    # Initliase model, loss function and optimiser
    ann_model = ANN(input_size, num_class).to(device)
    ann_optimiser = torch.optim.Adam(ann_model.parameters(), lr=learning_rate)

    cnn_model = CNN(num_class).to(device)
    cnn_optimiser = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    # Train ANN
    print("\nTraining ANN Model:")
    ann_train_losses, ann_val_losses, ann_train_accuracies, ann_val_accuracies = train_model(ann_model, ann_optimiser, train_loader, val_loader, loss_fn, num_epochs, device)

    print("\nTraining CNN Model:")
    cnn_train_losses, cnn_val_losses, cnn_train_accuracies, cnn_val_accuracies = train_model(cnn_model, cnn_optimiser, train_loader, val_loader, loss_fn, num_epochs, device)

    # Plot ANN metrics
    plot_metrics(ann_train_losses, ann_val_losses, ann_train_accuracies, ann_val_accuracies, num_epochs)

    # Plot CNN metrics
    plot_metrics(cnn_train_losses, cnn_val_losses, cnn_train_accuracies, cnn_val_accuracies, num_epochs)

    # Evaluate models
    print("\nEvaluating Models:")
    ann_accuracy = evaluate_model(ann_model, test_loader, device)
    cnn_accuracy = evaluate_model(cnn_model, test_loader, device)
    print(f"Final ANN Test Accuracy: {ann_accuracy:.2f}% \nFinal CNN Test Accuracy: {cnn_accuracy:.2f}%")

    # Save models
    os.makedirs("model", exist_ok=True)
    torch.save(ann_model.state_dict(), "model/ann_model.pth")
    torch.save(cnn_model.state_dict(), "model/cnn_model.pth")
