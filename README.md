# BiLSTM SentimentAnalysis : Sentiment Analysis of Movie Reviews Using Uni and Bi Directional LSTM

PROJECT STATEMENT

Deep Learning structure, consisting of a word embedding layer, a LSTM layer and a classification layer, to perform sentiment classification on movie review domain.

Introduction
PyTorch: https://pytorch.org/
GloVe: https://nlp.stanford.edu/projects/glove/
▪ I strongly recommend, also recommended by pytorch.org, to use Anaconda as the package manager since it installs all dependencies.
▪ You may also need other prerequisite packages, such as numpy, and you can simply use pip install numpy to install them under Anaconda.
▪ The package on e-learning is tested using Windows (OS), Python 3.7 (Language), None (CUDA).
▪ The package uses glove.840B.300d.txt as the pre-trained GloVe word vectors

Package
The package contains the following elements:
▪ Dataset
o GloVe
- Glove.840B.300d.txt : Not included, you need to download it from internet, unzip it and add it to the directory.
o Stsa
- Label.dev -> validation labels
- Label.test -> testing labels
- Label.train -> training labels 
- S1.dev -> validation data
- S1.test -> testing data
- S1.train -> training data

▪ Savedir

o Model.pickle -> save the best model on validation data 

▪ Data.py -> data processing

▪ Models.py -> main structure of the model

▪ Multis.py -> Utils class

▪ Train_nli.py -> training process of the model
