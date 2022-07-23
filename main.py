import data_loader_function
import data_loader_function_newTrial
import main_architecture
import train_test_model
import torch.nn as nn


News_LSTM = main_architecture.LSTM_news_classifier(50, 64, 7)

train_loader, val_loader, test_loader = data_loader_function_newTrial.data_loader_new()
# uncomment to train
train_test_model.train_net(News_LSTM, 128, 0.01, 100, 0.9, train_loader, val_loader, 'July_23_5_46')
# test code
# parameters = (50, 64, 7)
# model_path = train_test_model.get_model_path("LSTM_1", 128, 0.01, 16, "July_8_8_33")
# test_result = train_test_model.test_model(main_architecture.LSTM_news_classifier, parameters, False, model_path, test_loader, nn.MSELoss())
# print("Correct: {0}\tTotal: {1}\tAccuracy: {2}\tLoss: {3}".format(test_result[0], test_result[1], test_result[2], test_result[3]))
