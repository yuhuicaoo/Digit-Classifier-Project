import torch
import torch.nn as nn

def train_model(model, optimiser, train_loader, val_loader, loss_fn, num_epochs, device):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_tloss = 0

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

            # Adjust learning weights
            optimiser.step()

            # Gather data
            running_tloss += t_loss.item()
        
        # Calculate training loss for epoch
        epoch_loss = running_tloss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation Phase
        model.eval()
        running_vloss = 0

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

        # Calculate validation loss for epoch
        val_epoch_loss = running_vloss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | "
        )

    return train_losses, val_losses

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
    print(f"Final test accuracy: {accuracy:.2f}%")
    return accuracy

