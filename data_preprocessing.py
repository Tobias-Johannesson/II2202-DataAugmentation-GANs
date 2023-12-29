import boto3
import os

import numpy as np
import pandas as pd

from "constants.py" import *

def sample_data_loader(sample_size: int=500):
    """
        Randomly select n samples from the DataFrame
    """

    data = data_loader()
    data_sample = data.sample(n=sample_size, axis=1) #ignore_index=True?
    return data_sample

def data_loader():
    data = pd.read_csv(DATA_PATH)
    return data

def download_csv_from_s3(bucket_name, s3_file_path, local_file_path):
    """
        ...
    """
    s3_client = boto3.client('s3', 
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    
    try:
        s3_client.download_file(bucket_name, s3_file_path, local_file_path)
        print(f"File downloaded successfully: {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")