import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
# matplotlib inline
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("test.csv")
train = train.dropna()
# train = pd.read_csv("/content/disk/MyDrive/Machine_Hack/train.csv")
# test = pd.read_csv("/content/disk/MyDrive/Machine_Hack/test.csv")
# sub = pd.read_csv("/content/disk/MyDrive/Machine_Hack/submission.csv")
# train.head()

# test.head()


train_texts = train['Review'].values.tolist()
train_labels = train['Sentiment'].values.tolist()
# test_texts = test['Review'].values.tolist()


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,random_state=42,stratify=train_labels)


import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from transformers import Trainer,TrainingArguments

model_name  = 'distilbert-base-uncased'


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased',num_labels=3)

train_encodings = tokenizer(train_texts, truncation=True, padding=True,return_tensors = 'pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True,return_tensors = 'pt')
# test_encodings = tokenizer(test_texts, truncation=True, padding=True,return_tensors = 'pt')


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
    

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)
# test_dataset = SentimentTestDataset(test_encodings)

from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    #recall = recall_score(y_true=labels, y_pred=pred)
    #precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred, average='weighted')

    return {"accuracy": accuracy,"f1_score":f1}

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

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=3)

trainer = Trainer(
    model=model,# the instantiated 🤗 Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=train_dataset,# training dataset
    eval_dataset=val_dataset , # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

# test['Sentiment'] = 0 
# test_texts = test['Review'].values.tolist() 
# test_labels = test['Sentiment'].values.tolist() 
# test_encodings = tokenizer(test_texts, truncation=True, padding=True,return_tensors = 'pt').to("cuda") 
# test_dataset = SentimentDataset(test_encodings, test_labels)
# preds = trainer.predict(test_dataset=test_dataset)


# probs = torch.from_numpy(preds[0]).softmax(1)

# predictions = probs.numpy()# convert tensors to numpy array

# newdf = pd.DataFrame(predictions,columns=['Negative_0','Neutral_1','Positive_2'])
# new_df.head()

# def labels(x):
#     if x == 0:
#         return 'Negative_0'
#     elif x == 1:
#         return 'Neutral_1'
#     else:
#         return 'Positive_2'

# results = np.argmax(predictions,axis=1)
# test['Sentiment'] = results
# test['Sentiment'] = test['Sentiment'].map(labels)
# test.head()


# import seaborn as sns
# sns.countplot(x='Sentiment',data=test)