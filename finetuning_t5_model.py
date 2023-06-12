# -*- coding: utf-8 -*-

!pip install -q sumeval==0.2.2

import numpy as np
import pandas as pd
import csv

!pip install simplet5

import re
import nltk

train_data = pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv', nrows=2000)
validation_data = pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv', nrows=400)
test_data = pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/test.csv', nrows=400)

train_data = train_data.drop(['id'], axis=1)
validation_data = validation_data.drop(['id'], axis=1)
test_data = test_data.drop(['id'], axis=1)

print(train_data.shape, validation_data.shape, test_data.shape)

#Null values
print(train_data['article'].isnull().sum())
print(train_data['highlights'].isnull().sum())
print(validation_data['article'].isnull().sum())
print(validation_data['highlights'].isnull().sum())
print(test_data['article'].isnull().sum())
print(test_data['highlights'].isnull().sum())

#duplicates
print(train_data.duplicated(subset= ['article', 'highlights']).sum())
print(validation_data.duplicated(subset= ['article', 'highlights']).sum())
print(test_data.duplicated(subset= ['article', 'highlights']).sum())

from simplet5 import SimpleT5

model_t5 = SimpleT5()
model_t5.from_pretrained(model_type="t5", model_name="t5-base")

# simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
train_data_t5 = train_data.rename(columns={"highlights":"target_text", "article":"source_text"})
train_data_t5 = train_data_t5[['source_text', 'target_text']]
# T5 model expects a task related prefix: since it is a summarization task, we have to add prefix "summarize: "
train_data_t5['source_text'] = "summarize: " + train_data_t5['source_text']

validation_data_t5 = validation_data.rename(columns={"highlights":"target_text", "article":"source_text"})
validation_data_t5 = validation_data_t5[['source_text', 'target_text']]
validation_data_t5['source_text'] = "summarize: " + validation_data_t5['source_text']

test_data_t5 = test_data.rename(columns={"highlights":"target_text", "article":"source_text"})
test_data_t5 = test_data_t5[['source_text', 'target_text']]
test_data_t5['source_text'] = "summarize: " + test_data_t5['source_text']

model_t5.train(train_df=train_data_t5,
            eval_df=validation_data_t5, 
            source_max_token_len=150, 
            target_max_token_len=50, 
            batch_size=8, max_epochs=3, use_gpu=True)

model_t5.load_model("t5","/kaggle/working/outputs/simplet5-epoch-2-train-loss-1.7706-val-loss-1.9661", use_gpu=True)

model_t5.predict(test_data_t5['source_text'][11])

test_data_t5 = test_data_t5[:100]
len(test_data_t5)

pred = []
for test in test_data_t5['source_text']:
    res = model_t5.predict(test)
    pred.append(res)

!pip install rouge

from rouge import Rouge
rouge = Rouge()

scores = rouge.get_scores(pred, list(test_data_t5['target_text']), avg=True)

scores