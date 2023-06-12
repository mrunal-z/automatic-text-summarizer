This project was for the course CS-286: Natural Language Processing.
# Automatic Text Summarization Techniques

This project focuses on implementing automatic text summarization using two different methods: extractive and abstractive summarization.

## Extractive Summarization
The extractive method in this project utilizes the TextRank algorithm.

## Abstractive Summarization

For abstractive summarization, three transformer models were employed, namely BERT2BERT pre-trained, BERT2BERT fine-tuned, and T5 fine-tuned. These models were fine-tuned using the [CNN-DailyMail dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail) available on Kaggle.

## Interactive GUI

An interactive GUI was created using Anvil app and Google Colab, providing a user-friendly interface for text summarization.

## Report

A report in IEEE format, containing all the information related to this project, is available in the **report.pdf** file.

## Code Files

The code files related to fine-tuning the models can be found in the following locations:
- **finetuning_bert_models.py**
- **finetuning_t5_model.py**

The code for the TextRank algorithm as well as for creating the Anvil GUI is located in the file **summarization_gui.py**.


