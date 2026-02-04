
# Debugging Exercise1 (Python File)


import csv
from typing import Set
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



# Convert fruit ID to fruit name

def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    """
    Convert a numeric fruit ID into a fruit name.
    """
    fruits_list = sorted(list(fruits))

    if fruit_id < 0 or fruit_id >= len(fruits_list):
        raise RuntimeError("Fruit ID does not exist")

    return fruits_list[fruit_id]


#  Swap x/y coordinates in bounding boxes

def swap(coords: np.ndarray) -> np.ndarray:
    """
    Swap x and y coordinates in bounding boxes.
    Format: [x1, y1, x2, y2, class_id]
    """
    coords = coords.copy()
    coords[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]
    return coords

# Plot precision–recall curve from CSV

def plot_data(csv_file_path: str):
    precision = []
    recall = []

    with open(csv_file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            precision.append(float(row["precision"]))
            recall.append(float(row["recall"]))

    plt.plot(precision, recall)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.show()



# Generator

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), 1, 28, 28)



# Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        return self.model(x)



#  Train GAN

def train_gan(batch_size=32, num_epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.MNIST(
        root=".",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    loss_fn = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_len = real_images.size(0)

            real_labels = torch.ones((batch_len, 1), device=device)
            fake_labels = torch.zeros((batch_len, 1), device=device)

         
            noise = torch.randn(batch_len, 100, device=device)
            fake_images = generator(noise)

            all_images = torch.cat((real_images, fake_images))
            all_labels = torch.cat((real_labels, fake_labels))

            discriminator.zero_grad()
            d_loss = loss_fn(discriminator(all_images), all_labels)
            d_loss.backward()
            opt_d.step()

            # Train generator
            noise = torch.randn(batch_len, 100, device=device)
            generator.zero_grad()
            g_loss = loss_fn(discriminator(generator(noise)), real_labels)
            g_loss.backward()
            opt_g.step()

        print(f"Epoch {epoch} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

    print("Training finished.")

#  Main (run tests)

if __name__ == "__main__":

    # Test id_to_fruit
    fruits = {"apple", "orange", "melon", "kiwi", "strawberry"}
    print("Fruit with ID 2:", id_to_fruit(2, fruits))

    # Test swap
    coords = np.array([
        [10, 5, 15, 6, 0],
        [5, 3, 13, 6, 1]
    ])
    print("Swapped coords:\n", swap(coords))


