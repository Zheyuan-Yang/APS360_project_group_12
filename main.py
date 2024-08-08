import data_loader
import model
import train_test_model
import torch.nn as nn
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='content')
    parser.add_argument('--exp_path', type=str, default='exp')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_ckpt', type=str, default='./models/model_LSTM_bidirectional')
    args = parser.parse_args()
    Train = not args.eval
    device = 'cpu'
    News_model = model.LSTM_news_classifier(50, 256, 7)
    train_loader, val_loader, test_loader = data_loader.data_loader(batch_size=args.bs, data_path=args.data_path)
    if Train:
        train_test_model.train_net(News_model, train_loader, val_loader, device=device, batch_size=args.bs,
                                   learning_rate=args.lr, num_epochs=args.num_epochs, exp_path=args.exp_path)
    else:
        parameters = (50, 256, 7) # (input size, hidden size, number of classes)
        model_path = os.path.join('models', 'model_LSTM_bidirectional')
        test_result = train_test_model.test_model(model.LSTM_news_classifier_bidirectional, parameters, False, model_path, test_loader, nn.MSELoss())
        print("Correct: {0}\tTotal: {1}\tAccuracy: {2}\tLoss: {3}".format(test_result[0], test_result[1], test_result[2], test_result[3]))
