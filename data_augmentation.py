from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#from torchvision.utils import save_image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

from pygan._torch.gan_image_generator import GANImageGenerator

from models import *

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

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data
        image_size = 224
        #print(one_hot_labels)
        #print(one_hot_labels.shape)
        num_classes = one_hot_labels.shape[1]

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

def balance_dataset_with_gan(X, y):
    batch_size = 32
    num_channels = 3
    num_classes = len(y.unique())
    image_size = 224
    latent_dim = 128 # Number of nodes used for the generator
    epochs = 4 

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Scale the pixel values to [0, 1] range
    X = X.numpy().astype("float32") / 255.0
    X = X.transpose(0, 2, 3, 1) # Move channel to last position
    y = keras.utils.to_categorical(y, num_classes)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    #print(f"Shape of training images: {X.shape}")
    #print(f"Shape of training labels: {y.shape}")

    generator_in_channels = latent_dim + num_classes
    discriminator_in_channels = num_channels + num_classes

    generator = get_generator_model(generator_in_channels)
    discriminator = get_discriminator_model(discriminator_in_channels)

    cond_gan = ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    cond_gan.fit(dataset, epochs=epochs)
    print("GAN Finished Training")

    # Generating the fake/syntethic images
    generated_data = generate_data(cond_gan, latent_dim, num_classes)
    
    plt.imshow(generated_data[0])
    plt.axis('off')  # Turn off axis numbers
    plt.show()
    plt.imshow(generated_data[1])
    plt.axis('off')  # Turn off axis numbers
    plt.show()

    return X, y

def generate_data(cond_gan, latent_dim, num_classes):
    first_number = 1
    second_number = 1
    num_interpolation = 9
    trained_gen = cond_gan.generator

    # Sample noise for the interpolation.
    interpolation_noise = tf.random.normal(shape=(1, latent_dim))
    interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
    interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))

    # Convert the start and end labels to one-hot encoded vectors.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # Calculate the interpolation vector between the two labels.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
        
    fake_images = fake * 255.0 # Scale up to full RGB
    converted_images = fake_images.astype(np.uint8)
    converted_images = tf.image.resize(converted_images, (224, 224)).numpy().astype(np.uint8)

    return converted_images