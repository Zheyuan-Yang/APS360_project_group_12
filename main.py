import data_loader_function
import data_loader_function_newTrial
import main_architecture
import train_test_model
import torch.nn as nn


News_Transformer = main_architecture.Transformer_news_classifier_1(50, 64, 7)
Train = True
train_loader, val_loader, test_loader = data_loader_function.data_loader()
if Train:
    train_test_model.train_net(News_Transformer, 128, 0.01, 200, train_loader, val_loader, 'Aug_3_5_24')
else:
    parameters = (50, 64, 7)
    model_path = train_test_model.get_model_path("Transformer_news_classifier_1", 128, 0.01, 16, "July_8_8_33")
    test_result = train_test_model.test_model(main_architecture.LSTM_news_classifier, parameters, False, model_path, test_loader, nn.MSELoss())
    print("Correct: {0}\tTotal: {1}\tAccuracy: {2}\tLoss: {3}".format(test_result[0], test_result[1], test_result[2], test_result[3]))
