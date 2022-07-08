import data_loader_function
import main_architecture
import train_test_model


use_cuda = True
News_LSTM = main_architecture.LSTM_news_classifier(50, 64, 7)

train_loader, val_loader, test_loader = data_loader_function.data_loader()
train_test_model.train_net(News_LSTM, 128, 0.01, 2, 0.9, train_loader, val_loader, 'July_8_8_33')
