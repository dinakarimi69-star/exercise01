import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

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
        output = self.model(x)
        return output.view(x.size(0), 1, 28, 28)



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

def train_gan(
    batch_size: int = 32,
    num_epochs: int = 100,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.MNIST(
        root=".",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # Show example real images
    real_samples, _ = next(iter(train_loader))
    fig = plt.figure()
    for i in range(16):
        sub = fig.add_subplot(4, 4, i + 1)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis("off")
    fig.suptitle("Real images")
    fig.tight_layout()
    display(fig)
    time.sleep(5)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    loss_function = nn.BCELoss()
    lr = 0.0001
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):

            real_samples = real_samples.to(device)
            # using actual batch size
            current_batch_size = real_samples.size(0)

            # Labels (FIXED)
            real_samples_labels = torch.ones((current_batch_size, 1), device=device)
            generated_samples_labels = torch.zeros((current_batch_size, 1), device=device)

            # Generate fake samples
            latent_space_samples = torch.randn((current_batch_size, 100), device=device)
            generated_samples = generator(latent_space_samples)

            # Train discriminator
            all_samples = torch.cat((real_samples, generated_samples))
            all_labels = torch.cat((real_samples_labels, generated_samples_labels))

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Train generator
            latent_space_samples = torch.randn((current_batch_size, 100), device=device)
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated,
                real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Show generated images (FIXED)
            if n == len(train_loader) - 1:
                title = (
                    f"Generated images\n"
                    f"Epoch: {epoch} "
                    f"Loss D: {loss_discriminator:.2f} "
                    f"Loss G: {loss_generator:.2f}"
                )
                samples = generated_samples.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, i + 1)
                    sub.imshow(samples[i].reshape(28, 28), cmap="gray_r")
                    sub.axis("off")
                fig.suptitle(title)
                fig.tight_layout()
                clear_output(wait=False)
                display(fig)

    print("Training finished.")


