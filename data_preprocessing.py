import boto3
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
from torchvision.transforms import ToTensor, transforms

from constants import *

def sample_data_loader(file_path, sample_size: int=1000):
    """
        Randomly select n samples from the DataFrame
    """

    data = pd.read_csv(file_path)
    return data.sample(n=sample_size, axis=0) #ignore_index=True?

def data_loader(file_path, sample_size: int=-1):
    if sample_size > 0: return sample_data_loader(file_path, sample_size)
    return pd.read_csv(file_path)

def download_csv_from_s3(bucket_name, s3_file_path, local_file_path):
    """
        ...
    """

    s3_client = boto3.client('s3', 
                             region_name="eu-north-1",
                             aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
                             aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
                            )
    
    try:
        s3_client.download_file(Bucket=bucket_name, Key=s3_file_path, Filename=local_file_path)
        print(f"File downloaded successfully: {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def download_csv_from_kaggle(kaggle_file_path, local_file_path):
    """
        ...
    """

    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    api_key = {
        "username": kaggle_username,
        "key": kaggle_key
    }
    kaggle_json_content = json.dumps(api_key)

    kaggle_path = Path('/home/codespace/.kaggle')
    os.makedirs(kaggle_path, exist_ok=True)
    kaggle_json_path = Path("/home/codespace/.kaggle/kaggle.json")
    os.makefile(kaggle_path, exist_ok=True)

    # Write the JSON content to kaggle.json
    with open(kaggle_json_path, 'w') as file:
        file.write(kaggle_json_content)
    os.chmod(kaggle_json_path, 600)  

    api = KaggleApi()
    api.authenticate()

    # Download the kaggle dataset to local directory and unzip
    api.dataset_download_files(kaggle_file_path, path=local_file_path, unzip=True)

def clean_data(data: pd.DataFrame):
    # Turn pandas into PyTorch
    y = data["label"]
    X = data.drop(['label'], axis=1) # Remove labels and index from image_data

    return X, y

def reshape_data(X, y):
    """ 
        ...
    """

    number_of_images = len(X)
    shaped_images = torch.Tensor(X.values).reshape(number_of_images, 28, 28, 3).permute(0, 3, 1, 2)
    image_labels = torch.Tensor(y.values).reshape(number_of_images)

    # Resize images from 28x28 into 224x224 for VGG architecture
    reshaped_images = _reshape(shaped_images)

    return reshaped_images, image_labels

def _reshape(dataset):
    new_h, new_w = 224, 224
    reshaped_images = torch.Tensor(len(dataset), 3, new_h, new_w)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor()
    ])

    for i, image in enumerate(dataset):
        img = transform(image)
        reshaped_images[i] = img

    return reshaped_images

def data_split(X, y, tp: int= 0.8):
    number_of_images = len(X)

    # Get the range from 0 to split number, and split number to end
    split_at = int(tp * number_of_images)
    X_train, X_test = X[0:split_at], X[split_at:number_of_images]
    y_train, y_test = y[0:split_at], y[split_at:number_of_images]

    return X_train, X_test, y_train, y_test