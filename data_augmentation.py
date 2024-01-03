from imblearn.over_sampling import SMOTE
import numpy as np
import torch

def balance_dataset_with_smote(X, y):
    """
    Balances the dataset by applying SMOTE to generate synthetic samples for minority classes.

    :param X: Array-like, shape = [n_samples, n_features]. Training vectors.
    :param y: Array-like, shape = [n_samples]. Target values.

    :return: X_resampled, y_resampled: The augmented dataset.
    """

    X_flattened = _flatten_images(X)
    smote = SMOTE()
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
