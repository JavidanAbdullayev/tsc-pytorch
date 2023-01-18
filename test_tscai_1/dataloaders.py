import pandas as pd
import numpy as np
import sklearn
import os
from importlib import resources
from test_tscai_1.utils import read_all_datasets

def DataLoader(dataset_names):
    return read_all_datasets(dataset_names)
