import torch
import torch.nn as nn
torch.manual_seed(42)

def train_model(model, optimiser, train_loader, val_loader, loss_fn, num_epochs, device):
    # intialise list to store metric values
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_tloss = 0
        running_tcorrect = 0
        total_train_samples = 0

        for train_images, train_labels in train_loader:
            # Moves the data to CPU / GPU depending on what device is
            train_images, train_labels = train_images.to(device), train_labels.to(device)

            # Zero gradients for each batch
            optimiser.zero_grad()

            # Make predictions for this batch
            t_out = model(train_images)

            # Compute loss and its gradients
            t_loss = loss_fn(t_out, train_labels)
            t_loss.backward()

            # Update learning weights
            optimiser.step()

            # Gather loss and accuracy data
            running_tloss += t_loss.item()
            _ , predicted = torch.max(t_out, 1)
            running_tcorrect += (predicted == train_labels).sum().item()
            total_train_samples += train_labels.size(0)
        
        # Calculate training loss and accuracy for current epoch
        epoch_loss = running_tloss / len(train_loader)
        epoch_accuracy = running_tcorrect / total_train_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation Phase
        model.eval()
        running_vloss = 0
        running_vcorrect = 0
        total_val_samples = 0

        # Disable gradient computation and reduce memory usage
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                # Moves the data to CPU / GPU depending on what device is
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                # Make predictions for this batch
                v_out = model(val_images)

                # Compute loss 
                v_loss = loss_fn(v_out, val_labels)

                # Gather data
                running_vloss += v_loss.item()
                _ , predicted = torch.max(v_out, 1)
                running_vcorrect += (predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        # Calculate validation loss for epoch
        val_epoch_loss = running_vloss / len(val_loader)
        val_epoch_accuracy = running_vcorrect / total_val_samples
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy*100:.2f}% | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_accuracy*100:.2f}% "
        )

    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            out = model(images)
            _, preds = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

