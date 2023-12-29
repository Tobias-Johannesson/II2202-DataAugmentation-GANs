import numpy as np 
import pandas as pd

from "data_preprocessing.py" import * 
from "constants.py" import *

def main():
    #data = sample_data_loader()
    bucket_name = 'ii2202-data-augmentation-gans'
    s3_file_path = "./hmnist_28_28_RGB.csv"
    local_file_path = './datasets/hmnist_28_28_RGB.csv'
    download_csv_from_s3(bucket_name, s3_file_path, local_file_path)

    data.head()

if __name__ == '__main__':
    print("Running")
    main()
    print("Complete")