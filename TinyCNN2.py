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

# ---- One-hot encode labels ----

y_train_oh = F.one_hot(y_train, num_classes=10).float()
y_test_oh = F.one_hot(y_test, num_classes=10).float()

# y_train = tensor([2, 0, 5])
# tensor([
  # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # class 2
  # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # class 0
  # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]   # class 5
# ])

# ---- DataLoaders ----

train_loader = DataLoader(TensorDataset(x_train, y_train_oh), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test_oh), batch_size=64)

# ---- Define PyTorch model (for conversion) ----

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

# ---- Instantiate and convert to CrypTen ----

pytorch_model = TinyCNN(n_classes=10)
dummy_input = torch.empty(x_train.shape)  # shape [N, 1, 8, 8]
crypten_model = crypten.nn.from_pytorch(pytorch_model, dummy_input)
crypten_model.encrypt()

# ---- Training function ----

def train_model(model, loader, epochs=10, lr=0.01):
    model.train()
    loss_fn = crypten.nn.CrossEntropyLoss()
    optimizer = crypten.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            x_enc = crypten.cryptensor(batch_x)
            y_enc = crypten.cryptensor(batch_y)

            optimizer.zero_grad()
            output = model(x_enc)
            loss = loss_fn(output, y_enc)
            loss.backward()
            optimizer.step()

            total_loss += loss.get_plain_text().item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.10f}")

# ---- Accuracy evaluation ----

def avg_test_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0

    for batch_x, batch_y in loader:
        x_enc = crypten.cryptensor(batch_x)
        out = model(x_enc).get_plain_text()
        pred = out.argmax(1)
        true = batch_y.argmax(1)
        correct += (pred == true).sum().item()
        total += batch_y.size(0)

    return correct / total

# ---- Run training and evaluation ----
train_model(crypten_model, train_loader, epochs=10, lr=0.01)
accuracy = avg_test_accuracy(crypten_model, test_loader)
print(f" ✅ Encrypted Test Accuracy: {accuracy * 100:.2f}%")
