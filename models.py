import torch
from torch import nn
import torchvision.models as models
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

            # ...
            layers.Dense(image_size * image_size * generator_in_channels),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((image_size, image_size, generator_in_channels)),
            
            # ...
            layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            # Final Conv2DTranspose layer to get to 224x224 size
            layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding="same"),
        ],
        name="generator",
    )

    return generator

def get_discriminator_model(discriminator_in_channels):
    image_size = 224

    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((image_size, image_size, discriminator_in_channels)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
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
    num_epochs = 2 # 10
    print_interval = 2

    # Create DataLoader
    #X_tensor = torch.tensor(X, dtype=torch.float32)
    #y_tensor = torch.tensor(y, dtype=torch.long)
    #dataset = TensorDataset(X_tensor, y_tensor)
    #data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0  # Initialize the running loss for this epoch

        # Your data loading and training code goes here
        for batch in range(len(X) % 32): # Read data in batches using a dataloader
          inputs, labels = X[batch*32:batch*32+32], y[batch*32:batch*32+32]
          optimizer.zero_grad()
          outputs = model(inputs)

          labels = labels.long() # criterion needs long
          loss = criterion(outputs, labels)
          loss.backward()

          optimizer.step()
          running_loss += loss.item()

        if epoch % print_interval == 0:
            print(f'Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}')

        # Average loss for the epoch
        batch_size = 32
        num_batches = -(-len(X) // batch_size)  # Ceiling division to handle the last batch
        epoch_loss = running_loss / num_batches

        print(f'Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}')