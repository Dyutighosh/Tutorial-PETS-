import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import crypten
from crypten import mpc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize CrypTen
crypten.init()
torch.set_num_threads(1)

# Load and prepare the data
X, y = load_digits(return_X_y=True)

# The sklearn Digits dataset contains digit images as 1D vectors
# We need to reshape them into 2D (8x8 images)
X = np.expand_dims(X.reshape((-1, 8, 8)), 1)  # Shape: [N, 1, 8, 8] (N: number of samples)

# Split data into train/test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# One-hot encode labels
y_train_oh = torch.nn.functional.one_hot(y_train, num_classes=10).float()
y_test_oh = torch.nn.functional.one_hot(y_test, num_classes=10).float()

# Create DataLoader
train_dataset = TensorDataset(x_train, y_train_oh)
test_dataset = TensorDataset(x_test, y_test_oh)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset)

# Define TinyCNN using CrypTen
class TinyCNN(crypten.nn.Module):
    """A very small CNN to classify the sklearn digits data-set."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        self.conv1 = crypten.nn.Conv2d(1, 8, 3, stride=1, padding=0)
        self.conv2 = crypten.nn.Conv2d(8, 16, 3, stride=2, padding=0)
        self.conv3 = crypten.nn.Conv2d(16, 32, 2, stride=1, padding=0)
        self.fc1 = crypten.nn.Linear(32, n_classes)

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.conv3(x)
        x = x.relu()
        x = x.flatten(1)
        x = self.fc1(x)
        return x

# Training function for the encrypted model
def train_model(model, train_loader, epochs=10, lr=0.01):
    model.train()
    loss_fn = crypten.nn.CrossEntropyLoss()
    optimizer = crypten.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            encrypted_x = crypten.cryptensor(batch_x)
            encrypted_y = crypten.cryptensor(batch_y)

            optimizer.zero_grad()
            output = model(encrypted_x)
            loss = loss_fn(output, encrypted_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.get_plain_text().item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.10f}")

# Evaluation function (average test accuracy)
def avg_test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for batch_x, batch_y in test_loader:
        encrypted_x = crypten.cryptensor(batch_x)
        output = model(encrypted_x).get_plain_text()
        pred = output.argmax(dim=1)
        true = batch_y.argmax(dim=1)
        correct += (pred == true).sum().item()
        total += batch_y.size(0)

    return correct / total

# Initialize and encrypt the model
plain_model = TinyCNN(n_classes=10)
encrypted_model = plain_model.encrypt()

# Train the model
train_model(encrypted_model, train_loader, epochs=25)

# Evaluate the model on the test data
accuracy = avg_test_accuracy(encrypted_model, test_loader)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
