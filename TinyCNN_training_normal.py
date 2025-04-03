# Encrypted Inference with Timing

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import crypten

# ---- Init CrypTen ----
crypten.init()
torch.manual_seed(42)

# ---- Load and reshape sklearn digits ----
X, y = load_digits(return_X_y=True)
X = np.expand_dims(X.reshape((-1, 8, 8)), 1)  # [N, 1, 8, 8]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---- Convert to tensors ----
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ---- One-hot encode labels for CrypTen later ----
y_test_oh = F.one_hot(y_test, num_classes=10).float()

# ---- DataLoaders ----
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

# ---- Define PyTorch model for training ----
class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, 2, stride=1, padding=0)
        self.fc1 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x

# ---- Train using PyTorch ----
model = TinyCNN(n_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

start_train = time.time()

model.train()
for epoch in range(10):
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(train_loader):.4f}")

train_duration = time.time() - start_train

# ---- Convert trained model to CrypTen ----
dummy_input = torch.empty(x_test.shape)
crypten_model = crypten.nn.from_pytorch(model, dummy_input)
crypten_model.encrypt()  # Now it's MPC-ready

# ---- Encrypted inference (CrypTen) ----
def avg_test_accuracy_crypten(model, x_test_tensor, y_test_oh_tensor):
    model.eval()
    correct = 0
    total = 0

    test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_oh_tensor), batch_size=64)

    start_infer = time.time()

    for batch_x, batch_y in test_loader:
        x_enc = crypten.cryptensor(batch_x)
        output = model(x_enc).get_plain_text()
        pred = output.argmax(1)
        true = batch_y.argmax(1)
        correct += (pred == true).sum().item()
        total += batch_y.size(0)

    inference_duration = time.time() - start_infer
    accuracy = correct / total

    return accuracy, inference_duration

# ---- Run encrypted inference
accuracy, inference_duration = avg_test_accuracy_crypten(crypten_model, x_test, y_test_oh)
print(f"\n✅ Encrypted Inference Accuracy: {accuracy * 100:.2f}%")
print(f"⏱️  Training Time: {train_duration:.2f} seconds")
print(f"⏱️  Inference Time: {inference_duration:.2f} seconds")
