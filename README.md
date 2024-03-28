# Evidence Unlearning with a Modified Autoencoder (MAE) for MNIST
This project introduces a novel approach to deep learning by implementing a Modified Autoencoder (MAE) that not only learns to reconstruct the MNIST dataset's handwritten digits but also incorporates a mechanism for unlearning specific features in a controlled manner. The architecture utilizes both a main network for reconstruction and auxiliary networks to facilitate the unlearning process, making it a versatile tool for exploring the dynamics of learning and unlearning in neural networks.

## Key Features
- Convolutional Autoencoder Core: At the heart of the MAE is a convolutional autoencoder that efficiently encodes and decodes 28x28 pixel grayscale images of handwritten digits, focusing on capturing essential features for high-fidelity image reconstruction.
- Selective Unlearning: The project showcases an innovative approach to unlearning, where the model iteratively adjusts to discard or diminish the representation of specific features or information, guided by auxiliary networks. This process is performed in a loop, enabling the main network to refine its encoding strategy continuously.
- Filtered Training Capability: Offers the flexibility to train the model on filtered subsets of the MNIST dataset based on digit labels, facilitating focused studies on how selective data influences learning and unlearning.
- Visualization of Outcomes: Includes tools to visualize the original images alongside their reconstructions and the effects of unlearning, providing intuitive insights into the model's performance and behavior.
- Adaptable Training Framework: The customizable training loop accommodates various experimental setups, logging performance metrics, and automatically saving the model configurations that achieve the best validation performance.

## Implementation Highlights
- MAE Architecture: The autoencoder is designed with convolutional layers for encoding and decoding, complemented by auxiliary Q-layers that assist in the unlearning process, offering a unique blend of reconstruction and selective feature omission.
- Dynamic Training and Validation Loop: The system employs a dynamic loop that splits the MNIST dataset into training and validation sets, optionally applying filters based on digit labels. This setup is crucial for evaluating the effectiveness of learning and unlearning across different data subsets.
- Efficient Data Handling: Utilizes PyTorch's DataLoader and Dataset classes for streamlined data management, ensuring efficient preprocessing and optional dataset filtering.
- Qualitative Evaluation Tools: Leverages matplotlib for generating comparative visuals between original digits and their autoencoder-generated reconstructions, as well as illustrating the impact of unlearning on the reconstructed images.

## Project Aim
This project aims to delve into the mechanisms of unlearning within neural networks, providing a framework for investigating how autoencoders can be guided to selectively forget or downplay specific features. By alternating between learning and unlearning, the model offers a fascinating glimpse into the potential for neural networks to adapt and refine their representations based on evolving criteria.

## Usage
- Prepare Data Loaders: Use get_mnist_loaders to set up data loaders for the MNIST dataset, with options for filtering.
Launch Training and Unlearning Loop: Initiate the process with train_and_validate, adjusting parameters like epochs and learning rate to suit your experimental design.
- Evaluate and Visualize: Apply visualize_reconstructions to assess the model's ability to reconstruct images and observe the effects of unlearning.
Designed for researchers and enthusiasts alike, this project opens new avenues for exploring the balance between learning and unlearning in artificial intelligence.
