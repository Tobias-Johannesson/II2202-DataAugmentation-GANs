import numpy as np 
import pandas as pd

from "data_preprocessing.py" import * 
from "constants.py" import *

def main():
    data = sample_data_loader()
    data.head()

if __name__ == '__main__':
    print("Running")
    main()
    print("Complete")