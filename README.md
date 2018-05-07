Text Documents Optical Character Recognition
=======

# 1. Required Packages and Credits
LSTM part are developed from https://github.com/weinman/cnn_lstm_ctc_ocr
<br>
You need following libraries to run the code:
* Tkinter
* Tensorflow
* Opencv
* Numpy

<br>

# 2. **word_detect**:
Word Detection from Sentence + Word Recognition with CNN Network  

## cnn_detect:  

* python cnn_detect.py

<br>

## lstm_detect:  

* Download the lstm model from google drive: "https://drive.google.com/file/d/1fvNFbq1PWx_7PE0B9MnBmWHPVAUy8zKI/view?usp=sharing"
* place the downloaded model folder at word_detect/lstm/
* python lstm_detect.py

<br>

# 3. **model_training**:
Training Code of Neural Network for Recognition purpose

## cnn_train: Train CNN network for recognition  

* Download Emnist Dataset from http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
* Move Downloaded dataset in "model_training/cnn_train/dataset/emnist"
* Use "gunzip" command to unzip all .gz files
* cd "model_training/cnn_train", then "python main.py"
* Use user interface for further operations
![Graphical User Interface](https://github.com/jim19930609/Handwriting-OCR/blob/master/gui.png "GUI")

<br>

## lstm_train: Train LSTM network for recognition  

* Follow the README.md document in lstm_train folder


