import torch
import torch.nn as nn
import time
import csv
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_model_path(name, batch_size, learning_rate, exp_path='exp'):
    folder_path = "{0}_bs{1}_lr{2}".format(name,
                                               batch_size,
                                               learning_rate)
    folder_path = os.path.join(exp_path, folder_path)
    return folder_path

def find_the_best_model(val_acc):
    cur_largest = -1
    cur_largest_epoch = -1
    for epoch in range(len(val_acc)):
        if(val_acc[epoch] > cur_largest):
            cur_largest = val_acc[epoch]
            cur_largest_epoch = epoch
    return cur_largest_epoch, cur_largest


def train_net(net, train_loader, val_loader, batch_size=128, learning_rate=0.01, num_epochs=100, device='cuda', exp_path='exp'):
    assert num_epochs > 0, "num_epochs must be an integer that is greater than 0"
    assert learning_rate > 0, "learning_rate must be greater than 0"
    torch.manual_seed(1000)
    net.to(device)
    res_path = get_model_path(net.name, batch_size, learning_rate, exp_path=exp_path)
    os.makedirs(os.path.join(res_path, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(res_path, 'logs'), exist_ok=True)
    writer = SummaryWriter(os.path.join(res_path, 'logs'))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    epochs, train_losses, train_acc, val_losses, val_acc = [], [], [], [], []
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):
        epochs.append(epoch)
        total, correct = 0, 0
        total_loss = 0
        for articles, labels in train_loader:
            articles = articles.to(torch.device(device))
            labels = labels.to(torch.device(device))
            out = net(articles)
            loss = criterion(out, labels)
            total_loss = total_loss + loss.item() * articles.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pred = torch.squeeze(out.max(1, keepdim=True)[1], 1)
            correct = correct + pred.eq(torch.argmax(labels, dim=1)).sum().item()
            total = total + articles.shape[0]
        train_acc.append(correct/total)
        train_losses.append(total_loss/total)
        writer.add_scalar('train/acc', train_acc[-1], epoch)
        writer.add_scalar('train/loss', train_losses[-1], epoch)

        val_correct = 0
        val_total_loss = 0
        val_total = 0
        for val_articles, val_labels in val_loader:
            val_articles = val_articles.to(torch.device(device))
            val_labels = val_labels.to(torch.device(device))
            val_out = net(val_articles)
            # print(val_imgs)
            val_pred = torch.squeeze(val_out.max(1, keepdim=True)[1], 1)
            val_correct = val_correct + val_pred.eq(torch.argmax(val_labels, dim=1)).sum().item()
            val_total = val_total + val_articles.shape[0]
            val_total_loss = val_total_loss + (criterion(val_out, val_labels)).item() * val_articles.shape[0]
        val_losses.append(val_total_loss/val_total) # Append the average loss
        val_acc.append(val_correct/val_total)
        writer.add_scalar('val/acc', val_acc[-1], epoch)
        writer.add_scalar('val/loss', val_losses[-1], epoch)

        # print("Epoch {0}:\ntraining accuracy: {1}\ttraining loss: {2}\tvalidation accuracy: {3}\tvalidation loss:{4}".format(epoch, train_acc[epoch], train_losses[epoch], val_acc[epoch], val_losses[epoch]))
        # print("Correct number of outputs in validation: {0}\tTotal number of outputs in validation: {1}\tTotal validation loss {2}".format(val_correct, val_total, val_total_loss))
        torch.save(net.state_dict(), os.path.join(res_path, 'ckpts', str(epoch).zfill(5) + '.pth'))
    end_time = time.time()
    print("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % (
    (end_time - start_time), ((end_time - start_time) / num_epochs)))

    best_epoch, best_epoch_acc = find_the_best_model(val_acc)
    print("The best epoch: {0}\tAccuracy:{1}".format(best_epoch, best_epoch_acc))

def test_model(net_type, hyperparameters, model_path, data_loader, criterion, device='cuda'):
    state = torch.load(model_path, map_location=torch.device(device))
    net = net_type(hyperparameters[0], hyperparameters[1], hyperparameters[2])
    net.load_state_dict(state)
    correct = 0
    total_loss = 0
    total = 0
    for articles, labels in data_loader:
        articles = articles.to(device)
        labels = labels.to(device)
        out = net(articles)
        pred = torch.squeeze(out.max(1, keepdim=True)[1], 1)
        correct = correct + pred.eq(torch.argmax(labels, dim=1)).sum().item()
        total = total + articles.shape[0]
        total_loss = total_loss + (criterion(out, labels)).item() * articles.shape[0]
    return correct, total, correct / total, total_loss / total
