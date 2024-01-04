import random

import numpy as np 
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from data_preprocessing import *
from data_augmentation import *
from models import *
from evaluation import *
from visuals import *
from constants import *

def main():
    download_data(option=0) # 0 no download, 1 AWS, 2 Kaggle API

    # Use the local file
    file_path = "./datasets/hmnist_28_28_RGB.csv"
    data = data_loader(file_path, sample_size=10000)
    print("Data loaded")
    #plot_samples(data)

    X, y = clean_data(data)
    X, y = reshape_data(X, y)
    print(f"Data reshaped")
    #plot_samples(X)

    # Data split and then data augmentation
    X_train, X_test, y_train, y_test = data_split(X, y, 0.8)
    #X_train, y_train = balance_dataset_with_smote(X_train, y_train)
    X_train, y_train = balance_dataset_with_gan(X_train, y_train)
    # WGAN
    print(f"Data is augmented/balanced")

    number_of_classes = len(y.unique())
    model = get_vgg_model(number_of_classes)
    print("Model ready for training")    
    training_loop(model, X_train, y_train)
    print("Done with training")
    auc_score = get_auc(model, X_test, y_test, number_of_classes)
    print(f"Model AUC is: {auc_score}")

if __name__ == '__main__':
    main()