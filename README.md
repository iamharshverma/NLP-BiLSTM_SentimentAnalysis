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
- Glove.840B.300d.txt # Not included, you need to download it from internet, unzip it and add it to the directory.
o Stsa
- Label.dev # validation labelsvalidation labels validation labelsvalidation labels validation labelsvalidation labelsvalidation labels validation labels validation labelsvalidation labels
- Label.test # testing labelstesting labelstesting labels testing labelstesting labels testing labels testing labels testing labels testing labels
- Label.train # training labels # training labels# training labels# training labels# training labels # training labels # training labels # training labels # training labels# training labels
- S1.dev # validation datavalidation data validation datavalidation data validation datavalidation datavalidation data validation datavalidation data
- S1.test # testing datatesting datatesting data testing datatesting data testing data testing data
- S1.train # training data # training data# training data# training data# training data # training data # training data # training data# training data

▪ Savedir
o Model.pickle # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data # save the best model on validation data# save the best model on validation data # save the best model on validation data# save the best model on validation data# save the best model on validation data # save the best model on validation data# save the best model on validation data
▪ Data.py # data pro data prodata pro data prodata processingcessing cessingcessing
▪ Models.py # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the # main structure of the modelmodel (need modification)
▪ Multis.py # not important to the project not important to the projectnot important to the projectnot important to the projectnot important to the project not important to the projectnot important to the project not important to the projectnot important to the projectnot important to the projectnot important to the project not important to the projectnot important to the project not important to the projectnot important to the projectnot important to the projectnot important to the project ^_^^_^
▪ Train_nli.py # training process of the modeltraining process of the modeltraining process of the modeltraining process of the modeltraining process of the model training process of the model training process of the model training process of the modeltraining process of the model training process of the model training process of the modeltraining process of the model training process of the modeltraining process of the modeltraining process of the model training process of the model (need modification)
