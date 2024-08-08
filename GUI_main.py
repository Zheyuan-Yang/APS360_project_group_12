from PyQt6.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt6 import uic
import sys
import torchtext
import torch
from model import LSTM_news_classifier, LSTM_news_classifier_bidirectional
import train_test_model
import pandas as pd
import os


def convert_text_to_tensor(text):
    text_list = text.lower().split()

    stopwords = []
    stop = open(os.path.join('content', 'stopwords.txt'), encoding="utf-8")
    for line in stop:
        stopwords.append(line.strip())
    stop.close()
    text_list_num = []
    glove = torchtext.vocab.GloVe(name='6B', dim=50)
    for word in text_list:
        if not (word.lower() in stopwords):
            text_list_num.append(glove[word.lower()])

    text_tensor = torch.zeros(1, len(text_list_num), 50)
    for i in range(len(text_list_num)):
        text_tensor[0][i] = text_list_num[i]

    return text_tensor


def test_model(input_tensor, model_type, parameters, model_path):
    state = torch.load(model_path, map_location=torch.device('cpu'))
    model = model_type(parameters[0], parameters[1], parameters[2])
    model.load_state_dict(state)
    out = model(input_tensor)
    return out


class NewsClassifierUI(QMainWindow):
    def __init__(self, info):
        super(NewsClassifierUI, self).__init__()
        uic.loadUi("mainwindow.ui", self)
        self.result = self.findChild(QLabel, "label")
        self.article = self.findChild(QTextEdit, "textEdit")
        self.button = self.findChild(QPushButton, "pushButton")
        self.result.setText("Enter the text and press the button")
        self.button.clicked.connect(self.classify)
        self.info = info
        self.categories = ["automobile", "entertainment", "politics", "science", "sports", "technology", "world"]
        df = pd.DataFrame(self.categories)
        self.dummy_array = pd.get_dummies(df).to_numpy()
        self.show()

    def classify(self):
        article_text = self.article.toPlainText()
        if len(article_text) == 0:
            return
        text_tensor = convert_text_to_tensor(article_text)
        out = test_model(text_tensor, self.info[0], self.info[2], self.info[1])
        pred = torch.squeeze(out.max(1, keepdim=True)[1], 1)[0].item()
        self.result.setText(self.categories[pred])



if __name__ == '__main__':
    News_classifier_app = QApplication(sys.argv)
    net_parameters = (50, 122, 7)
    model_path = os.path.join('models', 'model_LSTM_3_bs244_lr0.01_epoch73_exercise_Aug_11_2_bidirectional_1_layers_hidden_size_122')
    info = (LSTM_news_classifier_bidirectional, model_path, net_parameters)
    News_classifier_ui = NewsClassifierUI(info)
    News_classifier_app.exec()
