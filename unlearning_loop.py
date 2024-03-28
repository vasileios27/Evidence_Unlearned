import torch
from mainAE import MAE  # Import your Modified Autoencoder model
from AuxiliaryAE import get_mnist_loaders, visualize_reconstructions2
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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


def unlearning_loop(num_epochs=10, lr=0.002, log_filename="unlearning_log.txt", path_mainAE ='path_to_model_1.pth', path_AuxiliaryAE ='path_to_model_2.pth'   ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mainAE, AuxiliaryAE = load_two_mae_models(path_mainAE, path_AuxiliaryAE)
    for param in AuxiliaryAE.parameters():
        param.requires_grad = False

    # If using a GPU, remember to move the models to the GPU after loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mainAE.to(device)
    AuxiliaryAE.to(device)

    # Prepare the data loaders
    train_loader, val_loader, data_loader= get_mnist_loaders(filter_by_digit=True)

    # Define loss function
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(mainAE.parameters(), lr=lr) 

    visualize_reconstructions2(mainAE, data_loader, device, "before", n_images=10)

    # evaluation phase for mainAE in filtered dataset
    mainAE.eval()  # Set the model to evaluation mode
    for epoch in range(0,1):
        val_loss = 0.0
        for data in train_loader:
            with torch.no_grad():
                for data in val_loader:
                    imgs, _ = data
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, imgs)
                    val_loss += loss.item() * imgs.size(0)

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.6f}')

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
            loss = 1/(criterion(outputA, outputB)  + 0.1e-7) # Example loss calculation
            
            # Backpropagate only through networkA
            loss.backward()
        optimizer.step()
    visualize_reconstructions2(mainAE, data_loader, device, "after", n_images=10)



if __name__ == "__main__":
    # Example usage
    unlearning_loop(
        num_epochs=10, 
        lr=0.002, 
        log_filename="unlearning_log.txt", 
        path_mainAE ='model_0.005.pth', 
        path_AuxiliaryAE ='Au_0.01.pth'   )
