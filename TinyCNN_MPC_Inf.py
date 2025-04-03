import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import crypten
from crypten import mpc

# Optional: disable GPU if CUDA is unavailable
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

# ---- MPC wrapper for CrypTen inference ----
@mpc.run_multiprocess(world_size=2)
def run_encrypted_inference():
    crypten.init()

    # ---- Load and preprocess sklearn digits ----
    X, y = load_digits(return_X_y=True)
    X = np.expand_dims(X.reshape((-1, 8, 8)), 1)  # [N, 1, 8, 8]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    y_test_oh = F.one_hot(y_test, num_classes=10).float()

    # ---- Train model in plaintext ----
    model = TinyCNN(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

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

    # ---- Convert to CrypTen model inside MPC ----
    dummy_input = torch.empty(x_test.shape)
    crypten_model = crypten.nn.from_pytorch(model, dummy_input)
    crypten_model.encrypt()
    crypten_model.eval()

    # ---- Perform encrypted inference ----
    start_infer = time.time()
    correct = 0
    total = 0
    test_loader = DataLoader(TensorDataset(x_test, y_test_oh), batch_size=64)
    for batch_x, batch_y in test_loader:
        x_enc = crypten.cryptensor(batch_x)
        out_enc = crypten_model(x_enc)
        out_plain = out_enc.get_plain_text()
        pred = out_plain.argmax(1)
        true = batch_y.argmax(1)
        correct += (pred == true).sum().item()
        total += batch_y.size(0)

    inference_duration = time.time() - start_infer
    accuracy = correct / total

    # ---- Print Results ----
    print(f"\n✅ MPC Encrypted Inference Accuracy: {accuracy * 100:.2f}%")
    print(f"⏱️  Training Time: {train_duration:.2f} seconds")
    print(f"⏱️  Inference Time: {inference_duration:.2f} seconds")

if __name__ == "__main__":
    run_encrypted_inference()
