import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Define the model architecture (same as before)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# 2. Load the model
model = Autoencoder(input_dim=784, encoding_dim=32)
model.load_state_dict(torch.load('undercomplete_autoencoder.pth'))
model.eval()  # Set to evaluation mode

# 3. Use it for inference
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

images, labels = next(iter(test_loader))
images_flat = images.view(images.size(0), -1)

with torch.no_grad():
    latent_space = model.encode(images_flat)  # Get 32-dim representation
    reconstructed = model(images_flat)  # Get reconstructed images

reconstructed = reconstructed.view(images.size(0), 1, 28, 28)

# Display
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 6))
for i in range(4):
    axes[0, i].imshow(images[i].numpy().squeeze(), cmap='gray')
    axes[0, i].set_title(f"Label: {labels[i].item()}")
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    axes[1, i].imshow(reconstructed[i].numpy().squeeze(), cmap='gray')
    axes[1, i].set_title(f"Reconstructed: {labels[i].item()}")
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)

# Display latent space as 4x8 grid
for i in range(4):
    latent_4x8 = latent_space[i].view(4, 8).numpy()
    axes[2, i].imshow(latent_4x8, cmap='viridis')
    axes[2, i].set_title('Latent Space (4x8)')
    axes[2, i].axis('off')

plt.tight_layout()    
plt.show()


print(f"Latent space shape: {latent_space.shape}")