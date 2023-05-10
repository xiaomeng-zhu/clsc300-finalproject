import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from transformers import Trainer,TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

ALL_AXIS = ["positivity","sincerity","concreteness","intensity"]
MODEL_NAME = 'distilbert-base-uncased'

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

## Test Dataset
class SentimentTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings)

def prepare_data(axis):
    all = pd.read_csv("labeled_data.csv")
    all = all.dropna()
    X = all["sequence"].values.tolist()
    y = all[axis].values.tolist()

    if axis == "positivity":
        num_labels = 3
    else:
        num_labels = 2
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=0.33, random_state=42, stratify = y)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME,num_labels=num_labels)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True,return_tensors = 'pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,return_tensors = 'pt')

    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    return train_dataset, val_dataset

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    #recall = recall_score(y_true=labels, y_pred=pred)
    #precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred, average='weighted')

    return {"accuracy": accuracy,"f1_score":f1}

def main():
    axis = "positivity"
    train_dataset, val_dataset = prepare_data(axis)
    training_args = TrainingArguments(
        output_dir='./res',          # output directory
        evaluation_strategy="steps",
        num_train_epochs=5,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs4',            # directory for storing logs
        #logging_steps=10,
        load_best_model_at_end=True,
    )

    if axis == "positivity":
        num_labels = 3
    else:
        num_labels = 2
    print(num_labels)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=num_labels)

    trainer = Trainer(
        model=model,# the instantiated 🤗 Transformers model to be trained
        args=training_args, # training arguments, defined above
        train_dataset=train_dataset,# training dataset
        eval_dataset=val_dataset , # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()