import cv2
from lstm.lstm_detect import lstm_detect, build_lstm
from cnn.cnn_detect import cnn_detect, build_cnn_network

class Solver:
  def __init__(self):
    self.lstm_network = build_lstm()
    self.cnn_network = build_cnn_network()

  def Read_Paragraph_Image(self, path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    self.paragraph = Paragraph(img)

  def Recognize(self, method="lstm"):
    self.paragraph.Sentence_separation()
    if method == "lstm":
      self.paragraph.Detect_lstm(self.lstm_network)
    elif method == "cnn":
      self.paragraph.Detect_cnn(self.cnn_network)
    else:
      print "Invalid Methods Specified. Supported Methods are:"
      print "method = \"lstm\" \n method = \"cnn\""
      raise NotImplementedError

    return self.paragraph.paragraph


class Paragraph:
  def __init__(self, img):
    self.img = img
    self.sentence_collect = []
    self.sentencelist = []
    self.paragraph = ""

  def Sentence_separation(self):
    ################################################
    #   Deploy Sentence Separation Algorithm Here  #
    ################################################
    img = self.img
    self.sentence_collect.append(Sentence(img))
    raise NotImplementedError

  def Detect_lstm(self, network):
    for sentence in self.sentence_collect:
      sentence.Detect_lstm(network)
      self.sentencelist.append(sentence.sentence)
    self.paragraph = "\n".join(self.sentencelist)

  def Detect_cnn(self, network):
    for sentence in self.sentence_collect:
      sentence.Detect_cnn(network)
      self.sentencelist.append(sentence.sentence)
    self.paragraph = "\n".join(self.sentencelist)


class Sentence:
  def __init__(self, img):
    self.wordlist = []
    self.sentence = ""
    self.img = img

  def Detect_lstm(self, network, limit_pixel=10, limit_char=60, target_h=300):
    img = self.img
    self.sentence, self.wordlist = lstm_detect(img, network, limit_pixel, limit_char, target_h)

  def Detect_cnn(self, network, limit_pixel=10, limit_char=5, limit_word_mult=12, target_h=300):
    img = self.img
    self.sentence, self.wordlist = cnn_detect(img, network)



