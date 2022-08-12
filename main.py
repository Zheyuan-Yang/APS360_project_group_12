import data_loader_function
import data_loader_function_newTrial
import main_architecture
import train_test_model
import torch.nn as nn


News_model = main_architecture.Transformer_news_classifier_2(50, 256, 7)
Train = True
train_loader, val_loader, test_loader = data_loader_function.data_loader()
if Train:
    train_test_model.train_net(News_model, 128, 0.01, 100, train_loader, val_loader, 'Aug_12_03_30_hidden_size_256')
else:
    parameters = (50, 256, 7) # (input size, hidden size, number of classes)
    model_path = train_test_model.get_model_path("LSTM_3", 256, 0.01, 16, "Aug_6_22_40_hidden_size_256")
    test_result = train_test_model.test_model(main_architecture.LSTM_news_classifier_3, parameters, False, model_path, test_loader, nn.MSELoss())
    print("Correct: {0}\tTotal: {1}\tAccuracy: {2}\tLoss: {3}".format(test_result[0], test_result[1], test_result[2], test_result[3]))
