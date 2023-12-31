import random

import numpy as np 
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt

from data_preprocessing import *
from models import *
from evaluation import *
from visuals import *
from constants import *

def main():
    # S3 download example
    #bucket_name = "ii2202-datasets"
    #s3_file_path = "hmnist_28_28_RGB.csv"
    #local_file_path = "./datasets/hmnist_28_28_RGB.csv"
    #download_csv_from_s3(bucket_name, s3_file_path, local_file_path)

    # Kaggle download example
    #kaggle_file_path = "kmader/skin-cancer-mnist-ham10000"
    #local_file_path = "./datasets"
    #download_csv_from_kaggle(kaggle_file_path, local_file_path)  

    # Use the local file
    file_path = "./datasets/hmnist_28_28_RGB.csv"
    #data = data_loader(file_path)
    data = sample_data_loader(file_path, sample_size=1000)
    #print(data.head())
    #plot_samples(data)

    X,y = clean_data(data)
    print(f"Original shape: {X.shape}")
    X, y = reshape_data(X, y)
    print(f"New shape: {X.shape}")
    X_train, X_test, y_train, y_test = data_split(X, y, 0.8)

    # Data augmentation here...

    number_of_classes = len(y.unique())
    model = get_vgg_model(number_of_classes)
    print("Model ready for training")    

    training_loop(model, X_train, y_train)
    print("Done with training")

    auc_score = get_auc(model, X_test, y_test, number_of_classes)
    print(f"Model AUC is: {auc_score}")

if __name__ == '__main__':
    main()