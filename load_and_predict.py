import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import torch
import torch.nn
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
# from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer
from transformers import Trainer,TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

MODEL_NAME = 'distilbert-base-uncased'

positivity_mapping = {
    0: "neutral",
    1: "positive",
    2: "negative" # originally -1 in the dataset
}
sincerity_mapping = {
    0: "sarcasm",
    1: "sincere",
}
concreteness_mapping = {
    0: "ambiguous",
    1: "concrete",
}
intensity_mapping = {
    0: "calm",
    1: "intense"
}

def arg_to_label(axis, arg):
    if axis == "positivity":
        return positivity_mapping[arg]
    elif axis == "sincerity":
        return sincerity_mapping[arg]
    elif axis == "concreteness":
        return concreteness_mapping[arg]
    elif axis == "intensity":
        return intensity_mapping[arg]
    else:
        print("No such axis exist")
    

def predict(text, axis):
    if axis == "positivity":
        num_labels = 3
    else:
        num_labels = 2

    model = DistilBertForSequenceClassification.from_pretrained("./ckpt/model-{}.pt/".format(axis))

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME,num_labels=num_labels)

    test_encodings = tokenizer([text], truncation=True, padding=True,return_tensors = 'pt')["input_ids"]
    # print(test_encodings)
    output = model(test_encodings)["logits"]
    # print(output)

    prediction = torch.argmax(output).item()
    # print(prediction)
    return arg_to_label(axis, prediction)

if __name__ == "__main__":
    res = predict("I hate and love you sooooo much", "concreteness")
    print(res)
    res = predict("I hate and love you sooooo much", "positivity")
    print(res)
    res = predict("I hate and love you sooooo much", "sincerity")
    print(res)
    res = predict("I hate and love you sooooo much", "intensity")
    print(res)
    