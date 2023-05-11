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
    X = [emoji.demojize(sequence)for sequence in X] # demojize all sequences
    y = all[axis].values.tolist()
    
    # clean up data labels
    updated_ys = []
    for label in y:
        if label == -1:
            updated_ys.append(2)
        else:
            updated_ys.append(int(label))
    y = updated_ys

    if axis == "positivity":
        num_labels = 3
    else:
        num_labels = 2
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=0.33, random_state=42, stratify = y)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME,num_labels=num_labels)
    # print(tokenizer.tokenize(train_texts[0]))

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

def main(axis):
    train_dataset, val_dataset = prepare_data(axis)
    training_args = TrainingArguments(
        output_dir='./res',          # output directory
        evaluation_strategy="steps",
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        # logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit = 1
    )

    if axis == "positivity":
        num_labels = 3
    else:
        num_labels = 2

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=num_labels)
    prev = model.state_dict()
    trainer = Trainer(
        model=model,# the instantiated 🤗 Transformers model to be trained
        args=training_args, # training arguments, defined above
        train_dataset=train_dataset,# training dataset
        eval_dataset=val_dataset , # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model('ckpt/model-{}.pt'.format(axis))


if __name__ == "__main__":
    # main("positivity")
    for axis in ALL_AXIS:
        print("===================================")
        print("Training Model on the dimension", axis)
        main(axis)