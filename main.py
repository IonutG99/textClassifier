import os
import re
import time

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
import pylatexenc.latex2text as latex2text
import re

translator = GoogleTranslator(source='ro', target='en')


def loadData(path):
    """" Function which loads the data from .txt files.
    :parameter path : path to the dataset
    :return: DataFrame with two columns, the Document text, and it's class
    """
    data = []
    for _, dirnames, _ in os.walk(path):
        for dirname in dirnames:
            for _, _, filenames in os.walk(os.path.join(path, dirname)):
                for filename in filenames:
                    if filename.endswith('.txt'):
                        filePath = os.path.join(path, dirname, filename)
                        with open(filePath, 'r', encoding='utf-8') as f:
                            dataText = f.read()
                        if dirname == 'calculus' and 'calculus' in filename or dirname == 'geometry':
                            data.append([latex2text.latex2text(dataText), dirname])
                        else:
                            data.append([dataText, dirname])
    textClassificationData = pd.DataFrame(data, columns=['Document', 'Class'])
    return textClassificationData


def encodeClasses(classes):
    """Map classes to numbers. For eg: ['sport','history','media'] -> {'sport':0,'history':1,'media':2}
    :parameter classes: classes as strings
    :returns dictionary as in example
    """
    encode = {item: classes.index(item) for item in classes}
    return encode


def splitTrainTest(data, percentage):
    """Function which splits the data into train and test data.
    :parameter data: data to split
    :parameter percentage: % how much of the data shall train and test
    :returns two Dataframes for train and test purpose
    """
    splitIndex = int(len(data) * percentage)
    return data[:splitIndex], data[splitIndex:]


def preprocess(text):
    """Function to preprocess the raw data, following 4 steps:
        - convert all document text to lowercase
        - remove all punctuation marks
        - remove all stopwords (e.g. 'the', 'is','and')
        - getting the root of each word using the Porter Stemmer algorithm
    :parameter text: raw string document
    :returns: preprocess document (also string)
    """
    text = text.lower()

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    porter_stemmer = PorterStemmer()
    text = ' '.join([porter_stemmer.stem(word) for word in text.split()])

    return text


class TfidfVectorizer:
    """ Class which implements one of the most popular techniques used to handle textual data. Tf-idf stands for term
    frequency-inverse document frequency. It is a statistical measure used to evaluate the importance of different
    words in a corpus(collection of documents). Importance i.e weight of tf-idf, is directly proportional to the
    number of times a word appears in a document (tf) and is inversely proportional to the frequency of that word
    across all the documents in the corpus.
    """

    def __init__(self):
        self._idf = None
        self._vocabulary = None

    def fit(self, X):
        """fit() method computes Inverse document frequency (IDF) for each word is calculated as the logarithm of the
        number of documents in the corpus divided by the total number of documents having that word in them. It also
        builds vocabulary of unique terms.  
        :param X: preprocessed documents
        """
        documentFrequency = Counter()
        for document in X:
            documentFrequency.update(set(document.split()))
        self._idf = np.log(len(X) / np.array(list(documentFrequency.values())))

        self._vocabulary = {term: i for i, term in enumerate(documentFrequency.keys())}

    def transform(self, X):
        """transform() function  will use features and idfs_ given to us by fit() and calculate tfidf values for all
        the features
        :param X: preprocessed documents
        :return: sparse matrix having the number of documents as total number of rows and features as columns
        """
        termFrequency = [Counter(document.split()) for document in X]
        data = []
        indices = []
        indptr = [0]
        for document in termFrequency:
            for term, freq in document.items():
                if term in self._vocabulary:
                    data.append(freq)
                    indices.append(self._vocabulary[term])
            indptr.append(len(indices))

        tfidf = csr_matrix(([f * self._idf[i] for i, f in zip(indices, data)],
                            indices,
                            indptr),
                           shape=(len(X), len(self._vocabulary)))
        return tfidf


class MultinomialNaiveBayes:
    """Class which implements from scratch a Multinomial Naive Bayes classifier."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._classes = None
        self._likelihoods = None
        self._priors = None

    def fit(self, X, y):
        """
        fit() method implements a train procedure for the classifier by defining its possible classes,
        the likelihoods for each word for each class by summing the occurrences of each word across all documents in
        that class and computes the prior probabilities.
        It also uses a Laplace smoothing which helps us to avoid 0/0 error.
        :param X: the train documents
        :param y: the train labels
        """
        self._classes = np.unique(y)
        self._likelihoods = np.zeros((X.shape[1], len(self._classes)))
        for i in range(len(self._classes)):
            self._likelihoods[:, i] = np.sum(X[y == i, :], axis=0)
        self._priors = self._classes / len(y)

        self._priors = self._priors + self.alpha
        self._likelihoods = self._likelihoods + self.alpha
        self._likelihoods /= np.sum(self._likelihoods, axis=0, keepdims=True)

    def predict(self, X):
        """
        predict() method implements a predict procedure by calculating log probabilities for each class.
        :param X:
        :return: index of class with the highest probability
        """
        log_probs = np.zeros((X.shape[0], len(self._classes)))
        for i in range(len(self._classes)):
            log_probs[:, i] = X.dot(np.log(self._likelihoods[:, i])) + np.log(self._priors[i])
        return np.argmax(log_probs, axis=1)


if __name__ == '__main__':
    # Load data
    textClassificationDataset = loadData(r'dataset_for_text_classification')
    # Shuffle the data
    textClassificationDataset = textClassificationDataset.sample(frac=1)
    # Encode the classes ( strings to numbers )
    encodedClasses = encodeClasses(list(set(textClassificationDataset['Class'])))
    unique_classes, class_counts = np.unique(textClassificationDataset['Class'], return_counts=True)
    textClassificationDataset['Class'] = [encodedClasses[x] for x in textClassificationDataset['Class']]
    # plot the bar chart
    fig, ax = plt.subplots()
    ax.bar(unique_classes, class_counts)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Entries')
    ax.set_xticks(unique_classes)
    ax.set_xticklabels(unique_classes)
    plt.show()
    # Create pie chart
    plt.pie(class_counts, labels=unique_classes, autopct='%1.1f%%', startangle=90)

    # Add title and legend
    plt.title('Class Distribution')
    plt.legend()

    # Show plot
    plt.show()

    # Split data into training and test sets
    trainData, testData = splitTrainTest(textClassificationDataset, 0.8)
    # Prepare training data
    trainTexts = trainData['Document']
    trainLabels = trainData['Class']

    # Prepare test data
    testTexts = testData['Document']
    testLabels = testData['Class']

    # Preprocess the train and test texts
    trainTexts = [preprocess(x) for x in trainTexts]
    testTexts = [preprocess(x) for x in testTexts]
    vectorizer = TfidfVectorizer()
    naive_bayes = MultinomialNaiveBayes()
    vectorizer.fit(trainTexts)
    trainTexts = vectorizer.transform(trainTexts)
    naive_bayes.fit(trainTexts, trainLabels)
    testTexts = vectorizer.transform(testTexts)
    y_pred = naive_bayes.predict(testTexts)
    y_pred = list(y_pred)
    testLabels = list(testLabels)
    conf_mat = confusion_matrix(y_pred, testLabels)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    print('Overall accuracy: {} %'.format(acc * 100))
    # Compute evaluation metrics
    precision = precision_score(testLabels, y_pred, average='macro')
    recall = recall_score(testLabels, y_pred, average='macro')
    f1 = f1_score(testLabels, y_pred, average='macro')
    accuracy = accuracy_score(testLabels, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlOrRd', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1))
    print("Accuracy: {:.2f}".format(accuracy))
    print("Enter/Paste your content into the text file. It can be either Latex or simply text.")
    print("Waiting 5 seconds for your input.")
    time.sleep(5)
    with open('input.txt', 'r') as f:
        USER_INP = f.read()
    if USER_INP != '':
        textsForClassification = USER_INP.split('<!====================================!>')
        if textsForClassification:
            stats = pd.DataFrame(columns=['Input Text', 'Class'])
            for item in textsForClassification:
                if item != '' or item != ' ':
                    try:
                        text = latex2text.latex2text(item)
                        print('Your input is \n' + text)
                        toPredict = [preprocess(text)]
                        toPredict = vectorizer.transform(toPredict)
                        prediction = naive_bayes.predict(toPredict)
                        realKey = [key for key, value in encodedClasses.items() if value == prediction[0]]
                        print(realKey[0])
                        stats = pd.concat(
                            [stats, pd.DataFrame({'Input Text': text, 'Class': realKey[0]}, index=[0])],
                            ignore_index=True)
                    except Exception as e:
                        print(e)
            stats.to_csv('stats.csv', encoding='utf-8')
