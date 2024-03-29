# -*- coding: utf-8 -*-
"""APS360.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19lFK1lUCVuCTEdwF9mq-YFX3ykQWwFjD
"""

import numpy as np
import pandas as pd
import torch
import torchtext

import matplotlib.pyplot as plt

# load data from csv file
fields = ['news_article', 'news_category']

train_data = pd.read_csv('/content/inshort_news_data-train.csv', header=0, encoding='ISO-8859-1', usecols=fields, skip_blank_lines=True)
val_data = pd.read_csv('/content/inshort_news_data-val.csv', header=0, encoding='ISO-8859-1', usecols=fields, skip_blank_lines=True)
test_data = pd.read_csv('/content/inshort_news_data-test.csv', header=0, encoding='ISO-8859-1', usecols=fields, skip_blank_lines=True)

print('Num training articles: ', len(train_data))
print('Num validation articles: ', len(val_data))
print('Num testing articles: ', len(test_data))

# Creating training and testing data
X_train = train_data['news_article']
Y_train = train_data['news_category']
"""
Y_train = np.zeros((X_train.shape[0],1))
for i in range((X_train.shape[0])):
  for j in range(7):
    if (train[j+1][i]==1):
      Y_train[i]=j
"""
X_test = test_data['news_article']
Y_test = test_data['news_category']
"""
Y_test = np.zeros((X_test.shape[0],1))
for i in range((X_test.shape[0])):
  for j in range(7):
    if (test[j+1][i]==1):
      Y_test[i]=j
"""
X_val = val_data['news_article']
Y_val = val_data['news_category']

print (X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

for i in range(X_train.shape[0]):
  X_train[i] = X_train[i].split()

for j in range(X_val.shape[0]):
  X_val[j] = X_val[j].split()

for k in range(X_test.shape[0]):
  X_test[k] = X_test[k].split()
    
Y_train = pd.get_dummies(Y_train)

print(X_train[0])
print(type(X_train))
print(X_train)

print(type(Y_train))
print(Y_train)

# to see what is the largest number of words in a article
# np.unique(np.array([len(ix) for ix in X_train]) , return_counts=True)
# np.unique(np.array([len(ix) for ix in X_val]) , return_counts=True)
np.unique(np.array([len(ix) for ix in X_test]) , return_counts=True)

# stopwords to eliminate useless words
stopwords = []
stop = open('/content/stopwords.txt', encoding="utf-8")
for line in stop:
  stopwords.append(line.strip())
stop.close()

# utilize Glove6B for embedding
glove = torchtext.vocab.GloVe(name='6B', dim=50)

# Filling the embedding matrix
embedding_matrix_train = np.zeros((X_train.shape[0], 61, 50))
embedding_matrix_val = np.zeros((X_train.shape[0], 61, 50))
embedding_matrix_test = np.zeros((X_test.shape[0], 61, 50))

for i in range(X_train.shape[0]):
  for j in range(len(X_train[i])):
    if not (X_train[i][j].lower() in stopwords):
      embedding_matrix_train[i][j] = glove[X_train[i][j].lower()]

for i in range(X_val.shape[0]):
  for j in range(len(X_val[i])):
    if not (X_val[i][j].lower() in stopwords):
      embedding_matrix_val[i][j] = glove[X_val[i][j].lower()]

for i in range(X_test.shape[0]):
  for j in range(len(X_test[i])):
    if not (X_test[i][j].lower() in stopwords):
      embedding_matrix_test[i][j] = glove[X_test[i][j].lower()]

