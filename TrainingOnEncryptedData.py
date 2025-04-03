import sys
import torch
import torchvision
import matplotlib.pyplot as plt


# python 3.7 is required
# assert sys.version_info[0] == 3 and sys.version_info[1] == 7, "python 3.7 is required"

import crypten
crypten.init()


digits = torchvision.datasets.MNIST(root='/tmp/data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

digits_test = torchvision.datasets.MNIST(root='/tmp/data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

plt.imshow(digits[0][0][0], cmap='gray', interpolation='none')
print("label for image is ", digits[0][1])

def take_samples(digits, n_samples=1000):
    """Returns images and labels based on sample size"""
    images, labels = [], []

    for i, digit in enumerate(digits):
        if i == n_samples:
            break
        image, label = digit
        images.append(image)
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), 10)
        labels.append(label_one_hot)

    images = torch.cat(images)
    labels = torch.stack(labels)
    return images, labels


images, labels = take_samples(digits, n_samples=100)

print(images.shape)
print(labels.shape)

images_enc = crypten.cryptensor(images)
labels_enc = crypten.cryptensor(labels)

images_enc[0]


# test set
images_test, labels_test = take_samples(digits_test, n_samples=20)
images_test_enc = crypten.cryptensor(images_test)
labels_test_enc = crypten.cryptensor(labels_test)


class LogisticRegression(crypten.nn.Module):
    
    def __init__(self):
        super().__init__()
        # images are 28x28 pixels
        self.linear = crypten.nn.Linear(28 * 28, 10)
        
    def forward(self, x):
    	x = x.view(x.size(0), -1)  # flatten each image from [28,28] to [784] -> CODE ADDED BY SOUMYADYUTI GHOSH: Reshape x to shape [batch_size, flattened_features]
    	return self.linear(x)

model = LogisticRegression().encrypt()

model(images_enc)

def train_model(model, X, y, epochs=10, learning_rate=0.05):
    criterion = crypten.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        print(f"epoch {epoch} loss: {loss.get_plain_text()}")
        loss.backward()
        model.update_parameters(learning_rate)
    return model

model = train_model(model, images_enc, labels_enc)

prediction = model(images_enc[3].unsqueeze(0))

prediction.get_plain_text().argmax()
plt.imshow(images[3], cmap='gray', interpolation='none')

def avg_test_accuracy(model, X, y):    
    output = model(X).get_plain_text().softmax(0)
    predicted = output.argmax(1)
    labels = y.get_plain_text().argmax(1)
    correct = (predicted == labels).sum().float()
    return float(correct / y.shape[0])

print(avg_test_accuracy(model, images_enc, labels_enc))

class CNN(crypten.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = crypten.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = crypten.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = crypten.nn.Dropout2d(0.25)
        self.dropout2 = crypten.nn.Dropout2d(0.5)
        self.fc1 = crypten.nn.Linear(9216, 128)
        self.fc2 = crypten.nn.Linear(128, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = x.max_pool2d(2)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # Flatten here -> Code added by me
        x = self.fc1(x)
        x = x.relu()
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


model = CNN().encrypt()

x = images_enc[0].unsqueeze(0)
print(x.shape)
model(x)

model = train_model(model, images_enc[:10, ], labels_enc[:10,], epochs=10)

prediction = model(images_enc[3].unsqueeze(0)).argmax()
prediction.get_plain_text().argmax()
plt.imshow(images[3], cmap='gray', interpolation='none')

print(avg_test_accuracy(model, images_enc, labels_enc))

# class CNN(crypten.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = crypten.nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = crypten.nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = crypten.nn.Dropout2d(0.25)
#         self.dropout2 = crypten.nn.Dropout2d(0.5)
#         self.fc1 = crypten.nn.Linear(9216, 128)
#         self.fc2 = crypten.nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = x.relu()
#         x = self.conv2(x)
#         x = x.relu()
#         x = x.max_pool2d(2)
#         x = self.dropout1(x)
#         x = self.fc1(x)
#         x = x.relu()
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x


import torch.nn as nn
import torch.nn.functional as F


class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


pytorch_model = PyTorchModel()
dummy_input = torch.empty(images.shape)
crypten_model = crypten.nn.from_pytorch(pytorch_model, dummy_input)

crypten_model.encrypt()

prediction = crypten_model(images_enc[3].unsqueeze(0))

print(prediction)
print(prediction.get_plain_text())
prediction.get_plain_text().argmax()


crypten_model = train_model(crypten_model, images_enc[:10, ], labels_enc[:10,], epochs=10)
prediction = model(images_enc[3].unsqueeze(0)).argmax()
prediction.get_plain_text().argmax()
plt.imshow(images[3], cmap='gray', interpolation='none')

print(avg_test_accuracy(model, images_enc, labels_enc))


# import torch.nn as nn
# import torch.nn.functional as F


# class PyTorchModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         #x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         #x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output