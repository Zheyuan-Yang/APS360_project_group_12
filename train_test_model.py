import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import csv


def get_model_path(name, batch_size, learning_rate, epoch, exercise_code):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}_exercise_{4}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch, exercise_code)
    path = "./model/" + path
    return path

def get_csv_path(name, batch_size, learning_rate, exercise_code):
    """ Generate a name for the csv file consisting of all training and validation data

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "data_{0}_bs{1}_lr{2}_exercise_{3}.csv".format(name,batch_size, learning_rate, exercise_code)
    path = "./model/" + path
    return path


def find_the_best_model(val_acc):
    """ Find the model with the best validation accuracy

    Args:
        validation accuracy list
    Returns:
        The epoch with the greatest accuracy and its accuracy
    """
    cur_largest = -1
    cur_largest_epoch = -1
    for epoch in range(len(val_acc)):
        if(val_acc[epoch] > cur_largest):
            cur_largest = val_acc[epoch]
            cur_largest_epoch = epoch
    return cur_largest_epoch, cur_largest


def save_to_csv(path, epochs, train_losses, train_acc, val_losses, val_acc, header):
    organized_data = []
    organized_data.append(header)
    for i in range(len(epochs)):
        organized_data.append([epochs[i], train_losses[i], train_acc[i], val_losses[i], val_acc[i]])
    f = open(path,'w+')
    write_csv = csv.writer(f)
    write_csv.writerows(organized_data)


def train_net(net, batch_size, learning_rate, num_epochs, momentum, train_loader, val_loader, exercise_code):
    assert num_epochs > 0, "num_epochs must be an integer that is greater than 0"
    assert learning_rate > 0, "learning_rate must be greater than 0"
    torch.manual_seed(1000)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    epochs, train_losses, train_acc, val_losses, val_acc = [], [], [], [], []
    start_time = time.time()
    for epoch in range(num_epochs):
        epochs.append(epoch)
        total, correct = 0, 0
        total_loss = 0
        for articles, labels in train_loader:
            #############################################
            #To Enable GPU Usage
            # if use_cuda and torch.cuda.is_available():
              # imgs = imgs.cuda()
              # labels = labels.cuda()
            #############################################
            #print(imgs)
            #print(labels)
            out = net(articles)
            loss = criterion(out, labels)
            total_loss = total_loss + loss.item() * articles.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(out.shape)
            pred = torch.squeeze(out.max(1, keepdim=True)[1], 1)
            # print(pred)
            # print(torch.argmax(labels, dim=1))
            correct = correct + pred.eq(torch.argmax(labels, dim=1)).sum().item()
            total = total + articles.shape[0]
            # print(correct, total)
        train_acc.append(correct/total)
        train_losses.append(total_loss/total)

        val_correct = 0
        val_total_loss = 0
        val_total = 0
        for val_articles, val_labels in val_loader:
            # if use_cuda and torch.cuda.is_available():
                # val_imgs = val_imgs.cuda()
                # val_labels = val_labels.cuda()
            val_out = net(val_articles)
            # print(val_imgs)
            val_pred = torch.squeeze(val_out.max(1, keepdim=True)[1], 1)
            val_correct = val_correct + val_pred.eq(torch.argmax(val_labels, dim=1)).sum().item()
            val_total = val_total + val_articles.shape[0]
            val_total_loss = val_total_loss + (criterion(val_out, val_labels)).item() * val_articles.shape[0]
        val_losses.append(val_total_loss/val_total) # Append the average loss
        val_acc.append(val_correct/val_total)

        print("Epoch {0}:\ntraining accuracy: {1}\ttraining loss: {2}\tvalidation accuracy: {3}\tvalidation loss:{4}".format(epoch, train_acc[epoch], train_losses[epoch], val_acc[epoch], val_losses[epoch]))
        print("Correct number of outputs in validation: {0}\tTotal number of outputs in validation: {1}\tTotal validation loss {2}".format(val_correct, val_total, val_total_loss))
        model_path = get_model_path(net.name, batch_size, learning_rate, epoch, exercise_code)
        torch.save(net.state_dict(), model_path)
    end_time = time.time()
    print("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % (
    (end_time - start_time), ((end_time - start_time) / num_epochs)))

    best_epoch, best_epoch_acc = find_the_best_model(val_acc)
    print("The best epoch: {0}\tAccuracy:{1}".format(best_epoch, best_epoch_acc))

    csv_path = get_csv_path(net.name, batch_size, learning_rate, exercise_code)
    header = ["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"]
    save_to_csv(csv_path, epochs, train_losses, train_acc, val_losses, val_acc, header)

    # plotting
    plt.title("Training Loss Curve")
    plt.plot(epochs, train_losses, label="Train")
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
    plt.show()

    plt.title("Training Accuracy Curve")
    plt.plot(epochs, train_acc, label="Training")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.show()


    plt.title("Validation Loss Curve")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
    plt.show()

    plt.title("Validation Accuracy Curve")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.show()


def test_model(net_type, parameters, use_cuda, model_path, data_loader, criterion):
    state = torch.load(model_path)
    net = net_type(parameters[0], parameters[1], parameters[2])
    net.load_state_dict(state)
    if use_cuda and torch.cuda.is_available():
        net.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')
    correct = 0
    total_loss = 0
    total = 0
    for articles, labels in data_loader:
        if use_cuda and torch.cuda.is_available():
            articles = articles.cuda()
            labels = labels.cuda()
        out = net(articles)
        pred = torch.squeeze(out.max(1, keepdim=True)[1], 1)
        correct = correct + pred.eq(torch.argmax(labels, dim=1)).sum().item()
        total = total + articles.shape[0]
        total_loss = total_loss + (criterion(out, labels)).item() * articles.shape[0]
    return correct, total, correct / total, total_loss / total
