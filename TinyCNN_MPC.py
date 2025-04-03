#Both Training and Testing in Encrypted MPC domain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import crypten
from crypten import mpc
from multiprocess_launcher import MultiProcessLauncher
import time

ALICE = 0
BOB = 1

@mpc.run_multiprocess(world_size=2)
def run_mpc_training():
    crypten.init()
    torch.manual_seed(42)

    # ---- Load and reshape sklearn digits ----
    X, y = load_digits(return_X_y=True)
    X = np.expand_dims(X.reshape((-1, 8, 8)), 1)  # [N, 1, 8, 8]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # ---- Convert to tensors ----
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # ---- One-hot encode labels ----
    y_train_oh = F.one_hot(y_train, num_classes=10).float()
    y_test_oh = F.one_hot(y_test, num_classes=10).float()

    # ---- Save encrypted tensors from respective parties ----
    crypten.save_from_party(x_train, "/tmp/x_train.pth", src=ALICE)
    crypten.save_from_party(y_train_oh, "/tmp/y_train.pth", src=BOB)
    crypten.save_from_party(x_test, "/tmp/x_test.pth", src=ALICE)
    crypten.save_from_party(y_test_oh, "/tmp/y_test.pth", src=BOB)

    # ---- Load encrypted tensors ----
    x_train_enc = crypten.load_from_party("/tmp/x_train.pth", src=ALICE)
    y_train_enc = crypten.load_from_party("/tmp/y_train.pth", src=BOB)
    x_test_enc = crypten.load_from_party("/tmp/x_test.pth", src=ALICE)
    y_test_enc = crypten.load_from_party("/tmp/y_test.pth", src=BOB)

    # ---- Define TinyCNN ----
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

    # ---- Instantiate and encrypt model ----
    model = TinyCNN(n_classes=10)
    dummy_input = torch.empty_like(x_train)
    crypten_model = crypten.nn.from_pytorch(model, dummy_input)
    crypten_model.encrypt()

    # ---- Train model ----
    def train_model(model, X, Y, epochs=10, lr=0.01):
        model.train()
        loss_fn = crypten.nn.CrossEntropyLoss()
        optimizer = crypten.optim.SGD(model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, Y)
            loss.backward()
            optimizer.step()
            print(f"[Epoch {epoch+1}] Loss: {loss.get_plain_text().item():.6f}")

        return model

    # ---- Accuracy ----
    def avg_test_accuracy(model, X, Y):
        model.eval()
        out = model(X).get_plain_text()
        pred = out.argmax(1)
        true = Y.get_plain_text().argmax(1)
        correct = (pred == true).sum().item()
        return correct / Y.size(0)

    start_time = time.time()
    crypten_model = train_model(crypten_model, x_train_enc, y_train_enc, epochs=10)
    train_duration = time.time() - start_time

    start_inference = time.time()
    accuracy = avg_test_accuracy(crypten_model, x_test_enc, y_test_enc)
    inference_duration = time.time() - start_inference

    print(f"⏱️  Training Time: {train_duration:.2f} seconds")
    print(f"⏱️  Inference Time: {inference_duration:.2f} seconds")

    print(f"✅  MPC Encrypted Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    run_mpc_training()
