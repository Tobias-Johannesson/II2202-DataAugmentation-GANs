from imblearn.over_sampling import SMOTE
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def balance_dataset_with_smote(X, y):
    """
    Balances the dataset by applying SMOTE to generate synthetic samples for minority classes.

    :param X: Array-like, shape = [n_samples, n_features]. Training vectors.
    :param y: Array-like, shape = [n_samples]. Target values.

    :return: X_resampled, y_resampled: The augmented dataset.
    """

    X_flattened = _flatten_images(X)
    _, counts = np.unique(y, return_counts=True)
    k = min(counts.min() - 1, 6)

    smote = SMOTE(k_neighbors=k) # Makes sure it works with small samples
    X_resampled_flat, y_resampled = smote.fit_resample(X_flattened, y)

    X_resampled = _reshape_to_image(X_resampled_flat)
    y_resampled = torch.from_numpy(y_resampled)
    return X_resampled, y_resampled

def _flatten_images(images):
    """
    Flattens a 4D array of images into a 2D array.

    :param images: 4D numpy array of shape (num_images, channels, height, width).
    :return: 2D numpy array of shape (num_images, height*width*channels).
    """

    num_images, channels, height, width = images.shape 
    return images.reshape(num_images, height * width * channels)

def _reshape_to_image(flattened_images, height: int=224, width: int=224, channels: int=3):
    """
    Reshapes a 2D array back into a 4D array of images.

    :param flattened_images: 2D numpy array of shape (num_images, height*width*channels).
    :param height: Original height of the images.
    :param width: Original width of the images.
    :param channels: Original number of channels in the images.
    :return: 4D numpy array of shape (num_images, channels, height, width).
    """
    
    num_images = flattened_images.shape[0]
    return torch.from_numpy(flattened_images).reshape(num_images, height, width, channels).permute(0, 3, 1, 2)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # nz: Input noise vector size, ngf: size of feature maps, nc: number of channels
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Size: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Size: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size: (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, nc, 4, 7, 1, bias=False),
            nn.Tanh()
            # Output Size: (nc) x 224 x 224
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # nc: number of channels, ndf: size of feature maps
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf) x 112 x 112

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*2) x 56 x 56

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*4) x 28 x 28

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*8) x 14 x 14

            nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # Output Size: 1 (real or fake)
        )

    def forward(self, input):
        return self.main(input)

def balance_dataset_with_gan(X, y):
    # Hyperparameters
    nz = 100  # Size of generator input (noise vector)
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    nc = 3    # Number of channels in the images (RGB)
    num_epochs = 10

    # Create the generator and discriminator
    netG = Generator(nz, ngf, nc)
    netD = Discriminator(nc, ndf)

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    real_label = 1
    fake_label = 0
    fixed_noise = torch.randn(64, nz, 1, 1)  # Noise for Generator

    # Create DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader, 0):
            # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            # Train with real
            batch_size = data[0].size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float)
            output = netD(data[0]).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake
            noise = torch.randn(batch_size, nz, 1, 1)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

    synthetic_images, synthetic_labels = _generate_synthetic_images(netG, y, nz)

    # Combine with original data
    X_balanced, y_balanced = _combine_datasets(X, synthetic_images, y, synthetic_labels)

    return X_balanced, y_balanced

def _generate_synthetic_images(generator, y, nz):
    # Implement function to generate synthetic images using the trained generator
    # Generate Synthetic Data
    # Determine the number of images to generate for each class
    # Generate images
    class_labels, class_counts = y.unique(return_counts=True)
    desired_count = max(class_counts.values())  # Target count for each class
    images_to_generate = {cls: desired_count - count for cls, count in class_counts.items() if count < desired_count}
    
    synthetic_images = []
    synthetic_labels = []

    for class_label, count in images_to_generate.items():
        for _ in range(count):
            noise = torch.randn(1, nz, 1, 1)  # Generate random noise as input for the GAN
            with torch.no_grad():
                fake_image = netG(noise).detach()
            
            synthetic_images.append(fake_image)
            synthetic_labels.append(class_label)  # Assign class label to the generated image

    # Convert lists to appropriate format (e.g., tensors or numpy arrays)
    synthetic_images = torch.cat(synthetic_images, 0)
    synthetic_labels = torch.tensor(synthetic_labels)

    return synthetic_images, synthetic_labels

def _combine_datasets(X, synthetic_images, y, synthetic_labels):
    # Implement function to combine original and synthetic data
    # Assuming 'X' and 'y' are your original data and labels
    X_combined = torch.cat([X, synthetic_images], 0)
    y_combined = torch.cat([y, synthetic_labels], 0)

    return X_combined, y_combined
