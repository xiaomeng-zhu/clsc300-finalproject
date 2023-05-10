import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from transformers import Trainer,TrainingArguments
warnings.filterwarnings('ignore')

all = pd.read_csv("labeled_data.csv")
X = all[["sequence"]]
y1 = all[["positivity"]]
y2 = all[["sincerity"]]
y3 = all[["concreteness"]]
y4 = all[["intensity"]]

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.33, random_state=42)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.33, random_state=42)
X_train, X_test, y3_train, y3_test = train_test_split(X, y3, test_size=0.33, random_state=42)
X_train, X_test, y4_train, y4_test = train_test_split(X, y4, test_size=0.33, random_state=42)

print(y1_train.head())
print(y1_test.head())