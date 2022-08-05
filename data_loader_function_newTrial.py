import numpy as np
import pandas as pd
import torch
import torchtext
from torch.utils.data import TensorDataset, DataLoader


def data_loader_new(batch_size=128):
    # load data from csv file
    fields = ['news_article', 'news_category']

    train_data = pd.read_csv('./content/inshort_news_data-train.csv', header=0, encoding='ISO-8859-1', usecols=fields,
                             skip_blank_lines=True)
    val_data = pd.read_csv('./content/inshort_news_data-val.csv', header=0, encoding='ISO-8859-1', usecols=fields,
                           skip_blank_lines=True)
    test_data = pd.read_csv('./content/inshort_news_data-test.csv', header=0, encoding='ISO-8859-1', usecols=fields,
                            skip_blank_lines=True)

    # Creating training and testing data
    X_train = train_data['news_article']
    Y_train = train_data['news_category']

    X_test = test_data['news_article']
    Y_test = test_data['news_category']

    X_val = val_data['news_article']
    Y_val = val_data['news_category']

    for i in range(X_train.shape[0]):
        X_train[i] = X_train[i].split()

    for j in range(X_val.shape[0]):
        X_val[j] = X_val[j].split()

    for k in range(X_test.shape[0]):
        X_test[k] = X_test[k].split()

    Y_train = pd.get_dummies(Y_train).to_numpy()
    Y_val = pd.get_dummies(Y_val).to_numpy()
    Y_test = pd.get_dummies(Y_test).to_numpy()

    # stopwords to eliminate useless words

    # utilize Glove6B for embedding
    glove = torchtext.vocab.GloVe(name='6B', dim=50)

    # Filling the embedding matrix
    embedding_matrix_train = np.zeros((X_train.shape[0], 61, 50))
    embedding_matrix_val = np.zeros((X_val.shape[0], 61, 50))
    embedding_matrix_test = np.zeros((X_test.shape[0], 61, 50))

    for i in range(X_train.shape[0]):
        for j in range(len(X_train[i])):
            embedding_matrix_train[i][j] = glove[X_train[i][j].lower()]

    for i in range(X_val.shape[0]):
        for j in range(len(X_val[i])):
            embedding_matrix_val[i][j] = glove[X_val[i][j].lower()]

    for i in range(X_test.shape[0]):
        for j in range(len(X_test[i])):
            embedding_matrix_test[i][j] = glove[X_test[i][j].lower()]

    X_train_t = torch.from_numpy(embedding_matrix_train).to(torch.float32)
    Y_train_t = torch.from_numpy(Y_train).to(torch.float32)
    X_val_t = torch.from_numpy(embedding_matrix_val).to(torch.float32)
    Y_val_t = torch.from_numpy(Y_val).to(torch.float32)
    X_test_t = torch.from_numpy(embedding_matrix_test).to(torch.float32)
    Y_test_t = torch.from_numpy(Y_test).to(torch.float32)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)

    print('Num training articles: ', len(train_dataset))
    print('Num validation articles: ', len(val_dataset))
    print('Num test articles: ', len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader