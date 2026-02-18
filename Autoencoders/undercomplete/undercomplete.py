import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
INPUT_DIM = 28 * 28  # MNIST images are 28x28
ENCODING_DIM = 32 # Size of the latent space

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root= './data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder,self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, encoding_dim) 
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = Autoencoder(INPUT_DIM, ENCODING_DIM)

# Loss and Optimizer
criterion = nn.MSELoss() # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    print(len(train_loader))
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)

        # Forward pass
        output = model(img)
        loss = criterion(output, img)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('.', end="") 
    print('')
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

# Visualization
test_dataset = datasets.MNIST(root= './data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Reconstruct
images_flat = images.view(images.size(0), -1)
with torch.no_grad():
    reconstructed = model(images_flat)
reconstructed = reconstructed.view(images.size(0), 1, 28, 28)

# Display
fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(12,4))
for i in range(4):
    axes[0, i].imshow(images[i].numpy().squeeze(), cmap='gray')
    axes[0, i].get_xaxis().set_visible(False)
    axes[0, i].get_yaxis().set_visible(False)
    axes[1, i].imshow(reconstructed[i].numpy().squeeze(), cmap='gray')
    axes[1, i].get_xaxis().set_visible(False)
    axes[1, i].get_yaxis().set_visible(False)
plt.show()

# save model
torch.save(model.state_dict(), 'undercomplete_autoencoder.pth')