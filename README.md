The nomenclature is as follows:
{dataset name}_{Model Architecture}

Project Descriptions:
Mnist_FC:
  Fully connected network with one hidden layer trained on the mnist dataset.
  Uses Functional API, which is overkill, but I wanted to try it out.

Fashion_Mnist_CNN:
  CNN trained on the fashion mnist dataset.
  Given is the architecture that performed well.
  Uses Sequential API.

Titanic_EDA:
  Ran EDA on the popular titanic dataset.
  Used all vanilla classifiers in scikit learn and tuned their hyperparameters to achieve an accuracy of 77%.
  Used pyTorch to create a Neural Net classifier and trained it.
  
TFNS_RNN:
  Ran Sentiment analysis on Twitter Financial News Sentiment dataset of ~10k training samples.
  Learned text tokenization and reverse lookup.
  Experimented with various architectures and embedding sizes.
  Used Vanilla RNN, LSTM and GRU to achieve a maximum test classification accuracy of 73.56%, 76.7%, and 76.98% respectively.
  
