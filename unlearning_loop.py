import torch
from mainAE import MAE  # Import your Modified Autoencoder model
from AuxiliaryAE import get_mnist_loaders, visualize_reconstructions2, write_to_file
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import numpy as np

def load_two_mae_models(model_path_1, model_path_2, input_shape=[28, 28, 1], filters=[28, 56, 112, 10, 20]):
    """
    Loads two instances of the MAE model from specified file paths.

    Parameters:
    - model_path_1: Path to the saved state dictionary for the first MAE instance.
    - model_path_2: Path to the saved state dictionary for the second MAE instance.
    - input_shape: The input shape of the images for the MAE model.
    - filters: The configuration of filters for the MAE model.

    Returns:
    - model_1: The first loaded MAE model instance.
    - model_2: The second loaded MAE model instance.
    """

    # Initialize two instances of the MAE model
    model_1 = MAE(input_shape=input_shape, filters=filters)
    model_2 = MAE(input_shape=input_shape, filters=filters)

    # Load the state dictionaries into the models
    model_1.load_state_dict(torch.load(model_path_1))
    model_2.load_state_dict(torch.load(model_path_2))

    return model_1, model_2

def evaluate_model(model, data_loader, criterion, device):
    """Evaluates the model on the given data loader and returns the average loss."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

def get_mnist_loaders(data_root='./data', batch_size=64, train_val_split_ratio=0.8):
    """
    Get MNIST data loaders, including a 'rest' loader for digits where digit % 3 != 0.

    Parameters:
    - data_root: Path to save/download the MNIST dataset.
    - batch_size: Batch size for the DataLoader.
    - train_val_split_ratio: Ratio to split training and validation data for digits where digit % 3 == 0.

    Returns:
    - train_loader: DataLoader for the training data (where digit % 3 == 0).
    - val_loader: DataLoader for the validation data (where digit % 3 == 0).
    - rest_loader: DataLoader for the data where digit % 3 != 0.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load the full MNIST training dataset
    full_train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    
    # Filter indices for train/val split where digit % 3 == 0 and digit % 3 != 0
    divisible_by_3_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label % 3 == 0]
    not_divisible_by_3_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label % 3 != 0]
    
    # Create subsets
    divisible_by_3_dataset = Subset(full_train_dataset, divisible_by_3_indices)
    not_divisible_by_3_dataset = Subset(full_train_dataset, not_divisible_by_3_indices)
    
    # Calculate sizes for splitting the divisible dataset into training and validation
    train_size = int(len(divisible_by_3_dataset) * train_val_split_ratio)
    val_size = len(divisible_by_3_dataset) - train_size
    
    # Split the divisible dataset
    train_dataset, val_dataset = torch.utils.data.random_split(divisible_by_3_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    rest_loader = DataLoader(not_divisible_by_3_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, rest_loader


def unlearning_loop(num_epochs=10, lr=0.002, log_filename="unlearning_log.txt", path_mainAE ='path_to_model_1.pth', path_AuxiliaryAE ='path_to_model_2.pth', beta= 0.1   ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mainAE, AuxiliaryAE = load_two_mae_models(path_mainAE, path_AuxiliaryAE)
    for param in AuxiliaryAE.parameters():
        param.requires_grad = False

    # If using a GPU, remember to move the models to the GPU after loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mainAE.to(device)
    AuxiliaryAE.to(device)


    # Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define the size of the validation set
    val_size = 10000  # For example, 10,000 images for validation
    train_size = len(train_dataset) - val_size  # The rest for training

    # Split the dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # DataLoader for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # Usually, you don't need to shuffle the validation set

    _, filter1loader, filter2loader = get_mnist_loaders(data_root='./data', batch_size=64, train_val_split_ratio=0.8)
    # Define loss function
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(mainAE.parameters(), lr=lr)
    best_loss = float('inf') 

    nameb = f'plots/UL/Before_l{lr}_b{beta}_reconstructions.png'        
    visualize_reconstructions3(mainAE, train_loader, device,nameb, n_images=10)
    rest_loss = evaluate_model(mainAE, filter2loader, criterion, device)
    filter_loss = evaluate_model(mainAE, filter1loader, criterion, device)
    write_to_file(log_filename, f'Before Unlearning loop for LR:{lr}')
    message = f"loss for filter : {filter_loss}\nloss for rest : {rest_loss}"
    write_to_file(log_filename, message)

    # evaluation phase for mainAE in filtered dataset
    # Set the model to evaluation mode

    for epoch in range(num_epochs):
        train_loss = 0.0
        for data in train_loader:
            imgs, _ = data
            imgs = imgs.to(device)            
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass through both networks (example)
            outputA = mainAE(imgs)
            with torch.no_grad():  # Ensures no gradients are computed for networkB
                outputB = AuxiliaryAE(imgs)
            
            # Compute loss using output from networkA (and possibly networkB, depending on your needs)
            loss = criterion(outputA,imgs) + beta * ( 1/(criterion(outputA, outputB)  + 0.1e-7)) # Example loss calculation
            
            # Backpropagate only through networkA
            loss.backward()
            optimizer.step()

                    # Validation phase
        mainAE.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                imgs, _ = data
                imgs = imgs.to(device)
                outputA = mainAE(imgs)
                outputB = AuxiliaryAE(imgs)
                # Compute loss using output from networkA (and possibly networkB, depending on your needs)
                loss = criterion(outputA,imgs) + 1/(criterion(outputA, outputB)  + 0.1e-7) # Example loss calculation
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
            path = f'models/modelsEvidenceUnlearning/UN_model_LR{lr}_b{beta}.pth'
            torch.save(mainAE.state_dict(), f'UN_model_{lr}.pth')
            best_loss = val_loss
            # Log training and validation loss
    namea = f'plots/UL/After_l{lr}_b{beta}_reconstructions.png'        
    visualize_reconstructions3(mainAE, train_loader, device,namea, n_images=10)
    rest_loss = evaluate_model(mainAE, filter2loader, criterion, device)
    filter_loss = evaluate_model(mainAE, filter1loader, criterion, device)
    message = f"After Evidence Unlearning\nloss for filter : {filter_loss}\nloss for rest : {rest_loss}\nBest_Epoch{best_epoch}\n"
    write_to_file(log_filename, message)
    

def visualize_reconstructions3(model, data_loader, device, path, n_images=10):
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
    plt.savefig(path)  # Save the plot
    plt.show()

if __name__ == "__main__":
    for lr in [0.005, 0.004, 0.003, 0.002, 0.0009, 0.006, 0.003]:
        unlearning_loop(
            num_epochs=100, 
            lr=lr, 
            log_filename=f"m0.004_A0.009_UN_log.txt", 
            path_mainAE ='model_0.004.pth', 
            path_AuxiliaryAE ='Au_0.0009.pth'   )