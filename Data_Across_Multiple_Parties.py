import sys
import torch
import torchvision
import matplotlib.pyplot as plt

# python 3.7 is required
# assert sys.version_info[0] == 3 and sys.version_info[1] == 7, "python 3.7 is required"

import crypten
from crypten import mpc
crypten.init()
torch.set_num_threads(1)

alice_data = torch.tensor([1, 2, 3.0])
bob_data = torch.tensor([4, 5, 6.0])

ALICE = 0
BOB = 1

@mpc.run_multiprocess(world_size=2)
def save_all_data():
    crypten.save(alice_data, "/tmp/data/alice_data.pth", src=ALICE)
    crypten.save(bob_data, "/tmp/data/bob_data.pth", src=BOB)
    
save_all_data()

@mpc.run_multiprocess(world_size=2)
def load_data():
    alice_data_enc = crypten.load("/tmp/data/alice_data.pth")
    bob_data_enc = crypten.load("/tmp/data/bob_data.pth")
    
    print(type(alice_data_enc))
    print(f"alice data is {alice_data_enc.get_plain_text()}")

load_data()

digits = torchvision.datasets.MNIST(root='/tmp/data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

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

images.shape

@mpc.run_multiprocess(world_size=2)
def save_digits():
    crypten.save(images, "/tmp/data/alice_images.pth", src=ALICE)
    crypten.save(labels, "/tmp/data/bob_labels.pth", src=BOB)
      
save_digits()