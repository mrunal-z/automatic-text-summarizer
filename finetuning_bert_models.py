# -*- coding: utf-8 -*-
"""FinalMonCopy of Bert Capstone.ipynb

### Importing Libraries
"""

import numpy as np
import pandas as pd
import csv

from google.colab import drive
drive.mount('/content/drive')

"""### Loading Data"""

train_data = pd.read_csv('/content/drive/MyDrive/cnn_dailymail/train.csv')
validation_data = pd.read_csv('/content/drive/MyDrive/cnn_dailymail/validation.csv')
test_data = pd.read_csv('/content/drive/MyDrive/cnn_dailymail/test.csv')

"""###Data Preprocessing"""

#dropping ID column
train_data = train_data.drop(['id'], axis=1)
validation_data = validation_data.drop(['id'], axis=1)
test_data = test_data.drop(['id'], axis=1)

print(train_data.shape, validation_data.shape, test_data.shape)

train_data.head()

validation_data.head()

test_data.head()

"""### BERT2BERT"""

!pip install datasets
import datasets
from datasets import Dataset

!pip install transformers==4.2.1
import transformers
from transformers import BertTokenizerFast

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(validation_data)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

batch_size=8 
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):
  
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()


  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

train_dataset = train_dataset.select(range(15000))

train_dataset = train_dataset.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "highlights"]
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

val_dataset = val_dataset.select(range(200))

val_dataset = val_dataset.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "highlights"]
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

val_dataset

from transformers import EncoderDecoderModel

bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
bert2bert.config.eos_token_id = tokenizer.eos_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

!pip install rouge_score

"""### FINE-TUNING"""

rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

  
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./",
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    overwrite_output_dir=True,
    save_total_limit=3,

)

trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()

"""###TESTING"""

!ls

dummy_bert2bert = EncoderDecoderModel.from_pretrained("./checkpoint-5500")

import datasets
from transformers import BertTokenizer, EncoderDecoderModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = EncoderDecoderModel.from_pretrained("./checkpoint-5500")
model.to("cuda")

test_dataset = Dataset.from_pandas(test_data)

test_dataset = test_dataset.select(range(200))

batch_size = 8

def generate_summary(batch):
 
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)


    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

pred_str = results["pred"]
label_str = results["highlights"]

rougeL_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid
rouge1_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
rouge2_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

print(rougeL_output)
print(rouge1_output)
print(rouge2_output)

results[1]

pred_str

test_dataset['article'][1]

"""### TESTING ON FINE-TUNED MODEL FROM HUGGING FACE"""

import datasets
from transformers import BertTokenizer, EncoderDecoderModel

model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail").to("cuda")
tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

test_dataset = Dataset.from_pandas(test_data)

test_dataset = test_dataset.select(range(200))

batch_size = 8

def generate_summary(batch):
 
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)


    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

pred_str = results["pred"]
label_str = results["highlights"]

rougeL_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid
rouge1_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
rouge2_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

print(rougeL_output)
print(rouge1_output)
print(rouge2_output)

results[1]

test_dataset['article'][1]

pred_str