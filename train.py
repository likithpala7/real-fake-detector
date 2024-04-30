import torch
from resnet_model import ResNet
from torch.utils.data import DataLoader
from data_loader import RealFakeDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

num_classes = 2  # Replace with the actual number of classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(num_classes=2).to(device) # Load the ResNet model
num_epochs = 5
batch_size = 16

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

# Define the data loaders
train_dataset = RealFakeDataset(data_folder='imagenet_ai_small/ADM/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = RealFakeDataset(data_folder='imagenet_ai_small/ADM/val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
losses = []
accuracies = []
val_losses = []
val_accuracies = []
for epoch in range(num_epochs):
    running_loss = 0.0
    accuracy = 0.0
    for i, data in enumerate(train_loader):
        # Get the inputs and labels from the data loader
        inputs, labels = data
        labels = labels.long()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accuracy += (outputs.argmax(1) == labels).sum()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            accuracy = accuracy/(batch_size*100)
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}, Accuracy: {accuracy:.4f}')
            losses.append(running_loss/100)
            accuracies.append(accuracy)
            running_loss = 0.0
            # Evaluate the model on the validation set
            model.eval()

            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data
                    labels = labels.long()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    val_accuracy += (outputs.argmax(1) == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy /= len(val_dataset)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Plot and save the training loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.savefig('training_plot.png')

# Plot and save the validation loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(val_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('validation_plot.png')

model_path = 'resnet_model.pth'
torch.save(model.state_dict(), model_path)
print('Training finished.')