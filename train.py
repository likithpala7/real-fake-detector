import torch
from resnet_model import ResNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_loader import RealFakeDataset

num_classes = 2  # Replace with the actual number of classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(num_classes=2).to(device) # Load the ResNet model
num_epochs = 10
batch_size = 16

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

# Define the data loaders
train_dataset = RealFakeDataset(data_folder='imagenet_ai_small/ADM/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
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
            running_loss = 0.0

model_path = 'resnet_model.pth'
torch.save(model.state_dict(), model_path)
print('Training finished.')