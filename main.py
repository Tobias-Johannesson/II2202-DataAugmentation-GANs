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

    sample_sizes = [500, 1000, 1500, 2000, 4000]

    # Use the local file
    metrics_path = 'vgg_metrics.json'
    sample_metrics = {}
    for sample_size in sample_sizes:
        sample_metrics[sample_size] = {'average_auc': [], 'average_accuracy': [], 'average_f1': [], 'average_recall': []}
        file_path = "./datasets/hmnist_28_28_RGB.csv"
        data = data_loader(file_path, sample_size=500)
        print("Data loaded")
        #plot_samples(data)

        X, y = clean_data(data)
        X, y = reshape_data(X, y)
        print(f"Data reshaped")
        #plot_samples(X)

        # Data split and then data augmentation
        X_train, X_test, y_train, y_test = data_split(X, y, 0.8)
        instances_before = X_train.shape[0]
        X_train, y_train = balance_dataset_with_smote(X_train, y_train)
        #X_train, y_train = balance_dataset_with_gan(X_train, y_train)
        instances_after = X_train.shape[0]
        print(f"Generated {instances_after - instances_before} new samples")
        # WGAN
        print(f"Data is augmented/balanced")

        number_of_classes = len(y.unique())
        model = get_vgg_model(number_of_classes)

        # Move it to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"On: {device}")
        model.to(device)
        X_train.to(device)
        y_train.to(device)
        print("Model ready for training")  

        training_loop(model, X_train, y_train)
        print("Done with training")

        metrics = get_metrics(model, X_test, y_test, number_of_classes)
        print("Evaluation Metrics:")
        print(f"Average AUC: {metrics['average_auc']:.2f}")
        print(f"Average Accuracy: {metrics['average_accuracy']:.2f}")
        print(f"Average F1 Score: {metrics['average_f1']:.2f}")
        print(f"Average Recall: {metrics['average_recall']:.2f}")

        sample_metrics[sample_size]['average_auc'].append(metrics['average_auc'])
        sample_metrics[sample_size]['average_accuracy'].append(metrics['average_accuracy'])
        sample_metrics[sample_size]['average_f1'].append(metrics['average_f1'])
        sample_metrics[sample_size]['average_recall'].append(metrics['average_recall'])

    # Save metrics to a file
    with open(metrics_path, 'w') as file:
        json.dump(sample_metrics, file)

    # To plot the saved metrics
    plot_metrics(metrics_path)

if __name__ == '__main__':
    main()