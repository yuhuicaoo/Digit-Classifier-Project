import torch
import torch.nn as nn
import os
from model import ANN, CNN
from train import train_model, evaluate_model
from get_data import load_data
from utils import visualise_samples, plot_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    input_size = 28 * 28
    num_class = 10
    num_epochs = 20
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
