1. word_detect:
  Word Detection from Sentence + Word Recognition with CNN Network

  1) cnn_detect:
    python cnn_detect.py

  2) lstm_detect:
    - Download the lstm model from google drive: "https://drive.google.com/file/d/1fvNFbq1PWx_7PE0B9MnBmWHPVAUy8zKI/view?usp=sharing"
    - place the downloaded model folder at word_detect/lstm/
    python lstm_detect.py

2. model_training:
  Training Code of Neural Network for Recognition purpose

  1) cnn_train: Train CNN network for recognition
    - Download Emnist Dataset from http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
    - Move Downloaded dataset in "model_training/cnn_train/dataset/emnist"
    - Use "gunzip" command to unzip all .gz files
    - cd "model_training/cnn_train", then "python main.py"

  2) lstm_train: Train LSTM network for recognition
    - Follow the README.md document in lstm_train folder


