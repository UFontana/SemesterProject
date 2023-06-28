# NLP Tools for Understanding the Role of Propaganda Techniques Online

This repository contains code and resources for classifying propaganda in textual content. The goal of this project is to develop a robust machine learning model capable of identifying and categorizing propaganda techniques present in news articles, social media posts, or any other form of written content. We perform tasks classification tasks on a sentence level. 

Specifically we peform two tasks:
1. binary task of detecting propaganda in a sentence
2.  mutlilabel classification task of classifying sentence into 18 classes based on detected propaganda techniques


The dataset containing fragment annotations of 18 propaganda techniques was obtained from the following paper: [Fine-Grained Analysis of Propaganda in News Articles](https://aclanthology.org/D19-1565.pdf)

The code is split in two folders for two corresponding models we used:

1. **fine_tuned_roberta_bert**:
      Fine tuning BERT and RoBERTa model for detection of propaganda in the sentence and multilabel classification into 18 propaganda technique classes
2. **sentence_bert_and_features**:
     Training traditional machine learning alghorithms for detection and multilabel classification using SentenceBERT embeddings and interpretable features.

