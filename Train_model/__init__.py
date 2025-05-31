import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader
from torchvision import transforms
from DatasetCat_Dog import Dataset_folder 
from Model import Model_Cat_Dog
from pathlib import Path

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
root = Path(__file__).parent.parent/"data"
# Dataset & Dataloader
train_dataset = Dataset_folder(root=str(root), train=True, transform=transform)
test_dataset  = Dataset_folder(root=str(root), train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model
model = Model_Cat_Dog().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Convert label text to int (nếu label là string "cat"/"dog")
        # Có thể bỏ qua nếu Dataset_folder đã trả về tensor
        labels = torch.tensor([0 if l == 'cat' else 1 for l in labels], dtype=torch.long).to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Training loss and accuracy
    epoch_loss = running_loss / len(train_dataset)
    accuracy = 100.0 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Evaluate on the test dataset after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Tắt gradient calculation trong evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * correct / total
    print(f"Test Accuracy after epoch {epoch+1}: {test_accuracy:.2f}%")

# Save the trained model
print("Training complete.")
torch.save(model.state_dict(), "best_model.pth")
