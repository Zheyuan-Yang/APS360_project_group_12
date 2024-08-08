# APS360 Group 12 Project: Use LSTM to classify news

## Introduction

   This is a APS360 course project. It aims to classify news into 7 categories using LSTM model. For details, please see the pdf report in this branch.

## Installation

   ```
   conda create -n NewsClassifier python=3.8
   pip install -r requirements.txt
   ```

## GUI Inference

   Run `python GUI_main.py`. Change line 70 to change the ckpt path.

## Train and evaluate

   Run `python main.py`.

   To evaluate a model, run `python main.py --eval --ckpt_path <path to a checkpoint>`

## Final report

   See APS360_Final_report.pdf