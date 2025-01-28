import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from model import ANN, CNN
from train import train_model, evaluate_model
from get_data import load_data

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.manual_seed(42)

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 100
    input_size = 28 * 28
    num_class = 10
    num_epochs = 10
    learning_rate = 0.001

    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size)

    # Initliase model, loss function and optimiser
    # model = ANN(input_size, num_class).to(device)
    cnn_model = CNN(num_class).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

    print("\nTraining Model:")
    train_loss, val_loss = train_model(cnn_model, optimiser, train_loader, val_loader, loss_fn, num_epochs, device)

    # Plot training and validation losses
    plt.figure(figsize=(10,6))
    plt.plot(range(1, num_epochs + 1), train_loss, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    print("\nEvaluating Model:")
    model_accuracy = evaluate_model(cnn_model, test_loader, device)

    os.makedirs("model", exist_ok=True)
    torch.save(cnn_model.state_dict(), "model/cnn_model.pth")
