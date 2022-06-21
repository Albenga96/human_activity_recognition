from utils.utils import basic_details

import inline as inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")


# Load datasets
train = pd.read_csv("data/train-1.csv")
test = pd.read_csv("data/test.csv")

train['Data'] = 'Train'
test['Data'] = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
both['subject'] = '#' + both['subject'].astype(str)

print(basic_details(both))
