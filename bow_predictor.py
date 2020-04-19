
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from datetime import datetime as dt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier




# stolen from https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
class GloveVectorizer:
  def __init__(self):
    # load in pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    embedding = []
    idx2word = []
    with open('../large_files/glove.6B/glove.6B.50d.txt') as f:
      # is just a space-separated text file in the format:
      # word vec[0] vec[1] vec[2] ...
      for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))

    # save for later
    self.word2vec = word2vec
    self.embedding = np.array(embedding)
    self.word2idx = {v:k for k,v in enumerate(idx2word)}
    self.V, self.D = self.embedding.shape

  def transform(self, sample):
    X = np.zeros(self.D)
    emptycount = 0
    tokens = sample.lower().split()
    vecs = []
    for word in tokens:
        if word in self.word2vec:
            vec = self.word2vec[word]
            vecs.append(vec)
    if len(vecs) > 0:
        vecs = np.array(vecs)
        #X = vecs.mean(axis=0)
        vecs = np.pad(vecs, ((0,3000-vecs.shape[0]), (0,0)), 'constant', constant_values=(0,0))
        X = vecs.flatten()
    return X

# returns news text for the day (cleaned sequence of words)
def get_days_news(date):
    MAX_FILE_SIZE = 65535
    month_str = {1 : 'January', 2 : 'February', 3 : 'March', 4 : 'April', 5 : 'May', 6 : 'June', 7 : 'July', 8 : 'August', 9 : 'September', 10 : 'October', 11 : 'November', 12 : 'December'}
    filename = "wpnews/" + str(date.day) + '_' + month_str[date.month] + '_' + str(date.year) + ".txt"
    f = open(filename, "r")
    text = f.read(MAX_FILE_SIZE)
    f.close()
    return text

# Returns (X_train, Y_train), (X_test, Y_test)
def get_input_data(test_fraction = 0.2, big_day_cutoff = 0.01):

    # Input samples are stored one to a file, named by date.
    # Input labels are collected in one CSV file.  
    # Some dates are missing in labels due to weekends and holidays.  
    # Just skip these dates.

    # read in labels first
    input_labels_file = "djia_change.csv"
    df_labels = pd.read_csv(input_labels_file)
    num_labels = len(df_labels)
    num_training_samples = int(num_labels * (1.0 - test_fraction))
    num_test_samples = num_labels - num_training_samples
    SAMPLE_SIZE = 3000 * 50
    X_train = np.zeros((num_training_samples, SAMPLE_SIZE))
    Y_train = np.zeros(num_training_samples)
    X_test = np.zeros((num_test_samples, SAMPLE_SIZE))
    Y_test = np.zeros(num_test_samples)

    # now iterate over labels and fetch corresponding news sample and store both
    labels = { 'bad day': 0, 'neutral': 1, 'good day': 2}
    sample_number = 1
    e = GloveVectorizer()

    for _, row in df_labels.iterrows():
        date = dt.strptime(row['date'], "%Y-%m-%d")
        if date.day == 29 and date.month == 2:
            continue
        change = row['change']
        if change <= -big_day_cutoff:
            label = labels['bad day']
        elif change >= big_day_cutoff:
            label = labels['good day']
        else:
            label = labels['neutral']

        sample = get_days_news(date)

        if sample_number > num_training_samples:
            X_test[sample_number - num_training_samples - 1, 0: SAMPLE_SIZE] = e.transform(sample)
            Y_test[sample_number - num_training_samples - 1] = label
        else:
            X_train[sample_number - 1, 0: SAMPLE_SIZE] = e.transform(sample)
            Y_train[sample_number - 1] = label

        sample_number += 1

    return (X_train, Y_train), (X_test, Y_test)



def main():
    (X_train, Y_train), (X_test, Y_test) = get_input_data()
    print('Training Samples:', X_train.shape)
    print('Training Labels:', Y_train.shape)
    print('Test Samples:', X_test.shape)
    print('Test Labels:', Y_test.shape)
    print('Training...')
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, Y_train)
    print("train score:", model.score(X_train, Y_train))
    print("test score:", model.score(X_test, Y_test))

if __name__ == "__main__":
    main()
