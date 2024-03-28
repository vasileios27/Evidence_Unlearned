import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import numpy as np

class MAE(nn.Module):
    """
    Modified Autoencoder (MAE) class definition.
    This class defines a denoising autoencoder with additional Q-layers for transfer learning purposes.
    """

    def __init__(self, input_shape=(28, 28, 1), filters=[28, 56, 112, 10, 20]):
        """
        Initialize the MAE.

        Parameters:
        - input_shape: A tuple representing the shape of the input images (height, width, channels).
        - filters: A list of filter sizes for each layer in the network.
        """
        super(MAE, self).__init__()
        self.input_shape = input_shape  # Store the input shape
        self.filters = filters  # Store the list of filter sizes

        # Padding styles for convolution layers
        pad3 = 'valid'
        pad4 = 'same'

        # Encoder Definition
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[2], filters[0], 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[1], 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=pad3),
            nn.ReLU(),
            nn.Flatten()
        )

        # First Dense Layer
        self.dense = nn.Sequential(
            nn.Linear(filters[2], filters[1]),
            nn.ReLU(),
            nn.Linear(filters[1], filters[3]),
            nn.ReLU()
        )

        # Transposed Dense Layer
        self.denseT = nn.Sequential(
            nn.Linear(filters[3], filters[1]),
            nn.ReLU(),
            nn.Linear(filters[1], filters[2]),
            nn.ReLU()
        )

        # Decoder Definition
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (filters[2], 1, 1)),
            nn.ConvTranspose2d(filters[2], filters[1], 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters[1], filters[0], 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters[0], input_shape[2], 6, stride=2)
        )

        # Q layers (for transfer learning, not used in initial training)
        self.Qlayer1 = nn.Sequential(
            nn.Linear(filters[2], filters[3]),
            nn.Softmax(dim=1)
        )
        self.Qlayer2 = nn.Sequential(
            nn.Linear(filters[2], filters[3]),
            nn.Softmax(dim=1)
        )
        self.Qlayer3 = nn.Sequential(
            nn.Linear(filters[2], filters[3]),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Defines the forward pass of the MAE.

        Parameters:
        - x: Input tensor.

        Returns:
        - x: Reconstructed output.
        - latZ: Latent space representation.
        - Qvec1, Qvec2, Qvec3: Outputs from the Q layers.
        """
        # Pass input through the encoder
        x = self.encoder(x)

        # Pass through dense and transposed dense layers
        x = self.dense(x)
        x = self.denseT(x)

        # Pass through the decoder to get the reconstructed output
        x = self.decoder(x)

        return x

    
def train_and_validate(model, train_loader, val_loader, device, num_epochs=1, criterion = nn.MSELoss(), lr=0.002,  log_filename="training_log.txt" ):
    write_to_file(log_filename, f"\nHyperparameters: lr={lr}, batch_size={train_loader.batch_size}")
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        for data in train_loader:
            imgs, _ = data
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                imgs, _ = data
                imgs = imgs.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                val_loss += loss.item() * imgs.size(0)

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)

        
        # Only print every 10 epochs (epoch numbers start from 0)
        if (epoch + 1) % 10 == 0:
         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
        
        # Save the model if validation loss has decreased
        if val_loss < best_loss:
            best_epoch = epoch
            print('Validation loss decreased ({:.6f} --> {:.6f}). in epoch {}.\nSaving model ...'.format(best_loss, val_loss,best_epoch))
            torch.save(model.state_dict(), f'Au_{lr}.pth')
            best_loss = val_loss
            # Log training and validation loss
        
    epoch_message = f'Epoch [{best_epoch}], Val Loss: {best_loss:.6f}\n'    
    write_to_file(log_filename, epoch_message)
    print("Training completed")



def visualize_reconstructions(model, data_loader, device, n_images=6):
    model.eval()  # Set the model to evaluation mode
    images, _ = next(iter(data_loader))  # Get a batch of images from the data loader
    images = images.to(device)
    with torch.no_grad():  # We do not need to compute gradients for visualization
        reconstructions = model(images)
    images = images.cpu()  # Move images and reconstructions to CPU for plotting
    reconstructions = reconstructions.cpu()
    
    plt.figure(figsize=(10, 4.5))

    for i in range(n_images):
        # Original Images
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed Images
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('Au_reconstructions.png')  # Save the plot

def visualize_reconstructions2(model, data_loader, device, lr, n_images=10):
    assert n_images <= 10, "n_images should be 10 or less"
    model.eval()  # Set the model to evaluation mode
    
    images_to_plot = {}  # To keep one image per label
    reconstructions_to_plot = {}  # To keep corresponding reconstructions
    
    with torch.no_grad():  # We do not need to compute gradients for visualization
        for images, labels in data_loader:
            images = images.to(device)
            reconstructions = model(images).cpu()
            images = images.cpu()
            
            for img, recon, label in zip(images, reconstructions, labels):
                if label.item() not in images_to_plot and len(images_to_plot) < n_images:
                    images_to_plot[label.item()] = img
                    reconstructions_to_plot[label.item()] = recon
                if len(images_to_plot) == n_images:
                    break
            if len(images_to_plot) == n_images:
                break
    
    plt.figure(figsize=(10, 4.5))
    labels_sorted = sorted(images_to_plot.keys())
    for i, label in enumerate(labels_sorted):
        # Original Images
        plt.subplot(2, n_images, i + 1)
        plt.imshow(images_to_plot[label].reshape(28, 28), cmap='gray')
        plt.title(f"Original {label}")
        plt.axis('off')

        # Reconstructed Images
        plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructions_to_plot[label].reshape(28, 28), cmap='gray')
        plt.title(f"Recon {label}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'plots/L{lr}Au_reconstructions.png')  # Save the plot
    plt.show()


def write_to_file(filename, message):
    with open(filename, "a") as file:  # Open in append mode
        file.write(message + "\n")  # Write message and a newline character

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

def get_mnist_loaders(data_root='./data', batch_size=64, train_val_split_ratio=0.8, filter_by_digit=None):
    """
    Get MNIST data loaders.
    
    Parameters:
    - data_root: Path to save/download the MNIST dataset.
    - batch_size: Batch size for the DataLoader.
    - train_val_split_ratio: Ratio to split training and validation data.
    - filter_by_digit: If provided, filters the dataset to only include digits where digit % 3 == 0.
    
    Returns:
    - train_loader: DataLoader for the training data.
    - val_loader: DataLoader for the validation data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load the full MNIST training dataset
    full_train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    
    data_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=False)

    if filter_by_digit is not None:
        # Filter indices by the given condition (e.g., labels divisible by 3)
        indices = [i for i, (_, label) in enumerate(full_train_dataset) if label % 3 == 0]
        full_train_dataset = Subset(full_train_dataset, indices)
    
    # Calculate sizes for splitting dataset into training and validation
    train_size = int(len(full_train_dataset) * train_val_split_ratio)
    val_size = len(full_train_dataset) - train_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, data_loader

# Now you can use train_loader for training and val_loader for validation or visualization.


def main(num_epochs=1, criterion=nn.MSELoss(), lr=0.002, log_filename="AUtraining_log.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the MAE
    model = MAE(input_shape=[28, 28, 1], filters=[28, 56, 112, 10, 20]).to(device)

# Usage example:
    train_loader, val_loader, data_loader = get_mnist_loaders(filter_by_digit=True)  # Set filter_by_digit to True to apply the y % 3 == 0 filter

    # Train and validate
    train_and_validate(model, train_loader, val_loader, device, num_epochs=num_epochs, criterion=criterion, lr=lr, log_filename=log_filename)
    
    # Visualization of reconstructions
    visualize_reconstructions2(model, data_loader, device, lr, n_images=10)

if __name__ == '__main__':
    for lr in [0.005, 0.004, 0.003, 0.002, 0.0009, 0.006, 0.003]:
        main(
            num_epochs=200, 
            criterion=nn.MSELoss(), 
            lr=lr, 
            log_filename="AU_log.txt")





