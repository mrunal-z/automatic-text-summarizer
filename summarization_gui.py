# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import nltk
import pickle
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import random
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from google.colab import drive
drive.mount('/content/drive')

import nltk
nltk.download('punkt')

nltk.download('stopwords')

!pip install datasets
import datasets
from datasets import Dataset

!pip3 install transformers==4.16.2

!pip install simplet5

!pip install anvil-uplink

from getpass import getpass
uplink_key = getpass('Enter your Uplink key: ')

import anvil.server
anvil.server.connect(uplink_key)

#t5 model

from simplet5 import SimpleT5

model_t5 = SimpleT5()
model_t5.from_pretrained(model_type="t5", model_name="t5-base")
model_t5.load_model("t5","/content/drive/MyDrive/finetuned-t5-model", use_gpu=True)

# preprocess the input text we get from user
@anvil.server.callable
def preprocess_input(text):
    ## clean 
    text = re.sub(r'[^\w\s]', '', str(text).strip())
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    return text

#generate summaries from t5 model
@anvil.server.callable
def generate_t5_summary(text):
  pred_txt = 'summarize: ' + text
  summary = model_t5.predict(pred_txt)
  trim_txt = summary[0].strip("'")
  return trim_txt

from transformers import BertTokenizer 
from transformers import EncoderDecoderModel

#bert2bert finetuned model
model_fine = EncoderDecoderModel.from_pretrained("/content/drive/MyDrive/bert-model").to("cuda")
tokenizer_fine = BertTokenizer.from_pretrained("bert-base-uncased")

#bert2bert cnn model
model_cnn = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail").to("cuda")
tokenizer_cnn = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

#function for generating bert summaries
def generate_summary(text, model, tokenizer):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str

#generate summaries for bert cnn
@anvil.server.callable
def generate_bert_summary_cnn(text):
  summary = generate_summary(text, model_cnn, tokenizer_cnn)
  trim_txt = summary[0].strip("'")
  return trim_txt


#generate summaries for bert finetuned
@anvil.server.callable
def generate_bert_summary_fine(text):
  summary = generate_summary(text, model_fine, tokenizer_fine)
  trim_txt = summary[0].strip("'")
  return trim_txt

#textrank algorithm

from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from scipy import spatial

@anvil.server.callable
def get_summary_text_rank2(text):
  # Split the text into lists of sentences and create a single list of all the sentences.
  sentences = []
  sentences.append(sent_tokenize(text))

  sentences = list(itertools.chain(*sentences))

  sentences_clean=[re.sub(r'[^\w\s]','',str(sentence).lower()) for sentence in sentences]
  stop_words = stopwords.words('english')
  sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]

  w2v=Word2Vec(sentences=sentence_tokens,min_count=1)
  sentence_embeddings=[[w2v.wv[word][0] for word in words] for words in sentence_tokens]
  max_len=max([len(tokens) for tokens in sentence_tokens])
  sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]

  similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
  for i,row_embedding in enumerate(sentence_embeddings):
      for j,column_embedding in enumerate(sentence_embeddings):
          similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)

  nx_graph = nx.from_numpy_array(similarity_matrix)
  scores = nx.pagerank_numpy(nx_graph)

  top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}
  top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:3])

  final_summ = ''
  for sent in sentences:
      if sent in top.keys():
          final_summ += sent
          
  return final_summ

# Link to Anvil App: 
# https://254MYUJYX36PBSOF.anvil.app/BNGRBOYLRGFKYSPIOKX6JFDB

#start server
anvil.server.wait_forever()