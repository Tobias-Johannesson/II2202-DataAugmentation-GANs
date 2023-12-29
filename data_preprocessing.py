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