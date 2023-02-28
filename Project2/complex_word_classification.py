"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Claudia Porto

I affirm that I have carried out my academic endeavors with full
academic honesty. Claudia Porto

Complete this file for parts 2-4 of the project.

"""
from collections import defaultdict
import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate


def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, labels = load_file(data_file)
    y_pred = [1] * len(words)
    evaluate(y_pred, labels)
    return y_pred


### 2.2: Word length thresholding

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)
    best_threshold = 0
    best_fscore = 0

    for threshold in range(1, 20):
        y_t_pred = [1 if len(word) >= threshold else 0 for word in t_words]
        y_d_pred = [1 if len(word) >= threshold else 0 for word in d_words]

        fscore = get_fscore(y_t_pred, t_labels)
        if fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold

    print(f"Best threshold: {best_threshold}")
    print(f"Best f-score: {best_fscore}")

    y_t_pred = [1 if len(word) >= best_threshold else 0 for word in t_words]
    y_d_pred = [1 if len(word) >= best_threshold else 0 for word in d_words]

    print('\nEvaluation Metrics on Training Data:')
    evaluate(y_t_pred, t_labels)
    print('\nEvaluation Metrics on Development Data:')
    evaluate(y_d_pred, d_labels)

    return y_t_pred, y_d_pred


### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt', encoding="utf8") as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)
    best_threshold = 0
    best_fscore = 0

    for threshold in range(40, 25000):
        y_t_pred = [1 if counts.get(word, 0) >= threshold else 0 for word in t_words]
        y_d_pred = [1 if counts.get(word, 0) >= threshold else 0 for word in d_words]

        fscore = get_fscore(y_t_pred, t_labels)
        if fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold}")
    print(f"Best f-score: {best_fscore}")

    y_t_pred = [1 if counts.get(word, 0) >= best_threshold else 0 for word in t_words]
    y_d_pred = [1 if counts.get(word, 0) >= best_threshold else 0 for word in d_words]

    print('\nEvaluation Metrics on Training Data:')
    evaluate(y_t_pred, t_labels)
    print('\nEvaluation Metrics on Development Data:')
    evaluate(y_d_pred, d_labels)

    return y_t_pred, y_d_pred


### 3.1: Naive Bayes

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)

    train_x = np.array([[len(word), counts.get(word, 0)] for word in t_words])
    dev_x = np.array([[len(word), counts.get(word, 0)] for word in d_words])
    train_y = np.array(t_labels)
    dev_y = np.array(d_labels)

    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)

    train_x_scaled = (train_x - mean) / sd
    dev_x_scaled = (dev_x - mean) / sd

    clf = GaussianNB()
    clf.fit(train_x_scaled, train_y)

    y_t_pred = clf.predict(train_x_scaled)
    y_d_pred = clf.predict(dev_x_scaled)

    print('\nEvaluation Metrics on Training Data:')
    evaluate(y_t_pred, train_y)
    print('\nEvaluation Metrics on Development Data:')
    evaluate(y_d_pred, dev_y)

    return y_t_pred, y_d_pred


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)

    train_x = np.array([[len(word), counts.get(word, 0)] for word in t_words])
    dev_x = np.array([[len(word), counts.get(word, 0)] for word in d_words])
    train_y = np.array(t_labels)
    dev_y = np.array(d_labels)

    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)

    train_x_scaled = (train_x - mean) / sd
    dev_x_scaled = (dev_x - mean) / sd

    clf = LogisticRegression()
    clf.fit(train_x_scaled, train_y)

    y_t_pred = clf.predict(train_x_scaled)
    y_d_pred = clf.predict(dev_x_scaled)

    print('\nEvaluation Metrics on Training Data:')
    evaluate(y_t_pred, train_y)
    print('\nEvaluation Metrics on Development Data:')
    evaluate(y_d_pred, dev_y)

    return y_t_pred, y_d_pred


### 3.3: Build your own classifier

def my_SVM_classifier(training_file, development_file):

    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)

    train_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in t_words])
    dev_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in d_words])
    train_y = np.array(t_labels)
    dev_y = np.array(d_labels)

    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)

    train_x_scaled = (train_x - mean) / sd
    dev_x_scaled = (dev_x - mean) / sd

    clf = SVC()
    clf.fit(train_x_scaled, train_y)

    y_t_pred = clf.predict(train_x_scaled)
    y_d_pred = clf.predict(dev_x_scaled)

    correct = set()
    incorrect = set()
    for i, word in enumerate(y_t_pred):
        if y_t_pred[i] != train_y[i]:
            incorrect.add(t_words[i])
        else:
            correct.add(t_words[i])

    print('Wrongly Classified Words:\n', incorrect)
    print('Correctly Classified Words:\n', correct)

    print('\nEvaluation Metrics on Training Data:')
    evaluate(y_t_pred, train_y)
    print('\nEvaluation Metrics on Development Data:')
    evaluate(y_d_pred, dev_y)

    return y_t_pred, y_d_pred


def my_Decision_Tree_classifier(training_file, development_file):

    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)

    train_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in t_words])
    dev_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in d_words])
    train_y = np.array(t_labels)
    dev_y = np.array(d_labels)

    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)

    train_x_scaled = (train_x - mean) / sd
    dev_x_scaled = (dev_x - mean) / sd

    clf = DecisionTreeClassifier()
    clf.fit(train_x_scaled, train_y)

    y_t_pred = clf.predict(train_x_scaled)
    y_d_pred = clf.predict(dev_x_scaled)

    print('\nEvaluation Metrics on Training Data:')
    evaluate(y_t_pred, train_y)
    print('\nEvaluation Metrics on Development Data:')
    evaluate(y_d_pred, dev_y)

    return y_t_pred, y_d_pred

def my_Random_Forest_classifier(training_file, development_file):

    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)

    train_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in t_words])
    dev_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in d_words])
    train_y = np.array(t_labels)
    dev_y = np.array(d_labels)

    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)

    train_x_scaled = (train_x - mean) / sd
    dev_x_scaled = (dev_x - mean) / sd

    clf = RandomForestClassifier()
    clf.fit(train_x_scaled, train_y)

    y_t_pred = clf.predict(train_x_scaled)
    y_d_pred = clf.predict(dev_x_scaled)

    print('\nEvaluation Metrics on Training Data:')
    evaluate(y_t_pred, train_y)
    print('\nEvaluation Metrics on Development Data:')
    evaluate(y_d_pred, dev_y)

    return y_t_pred, y_d_pred


def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_SVM_classifier(training_file, development_file)

if __name__ == "__main__":
    training_file = "C:/Users/claud/Desktop/CSC-483/Project2/data/complex_words_training.txt"
    development_file = "C:/Users/claud/Desktop/CSC-483/Project2/data/complex_words_development.txt"
    test_file = "C:/Users/claud/Desktop/CSC-483/Project2/data/complex_words_test_unlabeled.txt"

    print("Loading ngram counts ...")
    ngram_counts_file = "C:/Users/claud/Desktop/CSC-483/Project2/ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    #training_words, training_labels = load_file(training_file)
    #development_words, development_labels = load_file(development_file)

    #combined_words = training_words + development_words
    #combined_labels = training_labels + development_labels

    #combined_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in combined_words])
    #combined_y = np.array(combined_labels)

    #mean = combined_x.mean(axis=0)
    #sd = combined_x.std(axis=0)

    #combined_x_scaled = (combined_x - mean) / sd

    #clf = SVC()
    #clf.fit(combined_x_scaled, combined_y)

    #test_words, _ = load_file(test_file)
    #test_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in test_words])
    #test_x_scaled = (test_x - mean) / sd

    #test_labels = clf.predict(test_x_scaled)

    #with open("test_labels.txt", "w") as file:
        #for label in test_labels:
            #file.write(str(label) + "\n")



