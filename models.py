import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from tensorflow import keras
from tensorflow.keras import layers

def get_vgg_model(out_features: int):
    """
        ...
    """

    pretrained_vgg_model = models.vgg19_bn(weights='DEFAULT')

    IN_FEATURES = pretrained_vgg_model.classifier[-1].in_features
    OUTPUT_DIM = out_features
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    pretrained_vgg_model.classifier[-1] = final_fc

    return pretrained_vgg_model

def get_generator_model(generator_in_channels):
    image_size = 224
    num_channels = 3

    generator = keras.Sequential(
        [   
            keras.layers.InputLayer((generator_in_channels,)),

            # Start with a Dense layer to create a small feature map
            layers.Dense(7 * 7 * 512),  # Increased depth
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, 512)),  # Reshape to a small spatial dimension
            
            # Gradually upscale the image using Conv2DTranspose layers
            layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding="same"),
            layers.ReLU(),
            layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same"),
            layers.ReLU(),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.ReLU(),
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"),
            layers.ReLU(),

            # Final Conv2DTranspose layer to reach the desired image size
            #layers.Conv2DTranspose(num_channels, (4, 4), strides=(2, 2), padding="same", activation="sigmoid"),
            layers.Conv2DTranspose(num_channels, (4, 4), strides=(2, 2), padding="same", activation="tanh"),
        ],
        name="generator",
    )

    return generator

def get_discriminator_model(discriminator_in_channels):
    image_size = 224

    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((image_size, image_size, discriminator_in_channels)),

            # ...
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            #layers.Flatten(),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1),
        ],
        name="discriminator",
    )

    return discriminator

def training_loop(model, X, y):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Define the number of epochs and the interval to save performance
    num_epochs = 4 # 10
    print_interval = 2

    # Create DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0  # Initialize the running loss for this epoch

        # Your data loading and training code goes here
        #for batch in range(len(X) % 32): # Read data in batches using a dataloader
        for inputs, labels in data_loader:
          #inputs, labels = X[batch*32:batch*32+32], y[batch*32:batch*32+32]
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels) # criterion needs long (lables)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

        # Average loss for the epoch
        epoch_loss = running_loss / len(data_loader)

        if epoch % print_interval == 0:
            print(f'Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}')