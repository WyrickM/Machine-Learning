########################################################################
# Mantz Wyrick
# Machine Learning Homework #4
#
# Naive Bayes classifier. addressing data imbalance with
#   1) undersampling the majority
#   2) oversampling the minority
#   3) weight the data based on class imbalance
#
########################################################################

from google.colab import drive

drive.mount("/content/gdrive")

import cv2
import numpy as np
import pandas as pd
import pprint
import math
import matplotlib.pyplot as plt
import random
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import stop_words
from sklearn.utils import class_weight, compute_sample_weight
from nltk.tokenize import TreebankWordTokenizer
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE


"""
  getData()
    returns: X, y
      X => List of strings of the SMS messages
      y => List of strings that are the labels of the SMS messages

  Reads the data from the given text file and then splits the textfile by the 
  new line character into a list of strings. Then I split the list of strings 
  into a list of lists of strings by the tab character to get the label in index 0
  and the SMS string in index 1.
"""


def getData():
    textFile = open(
        "/content/gdrive/My Drive/WSU/CPTS/CPTS 437/Colab Notebooks/HW4/SMSSpamCollection",
        "r",
    )
    data = textFile.read().split("\n")
    X = [""] * (len(data) - 1)
    y = [""] * (len(data) - 1)
    for i in range(len(data) - 1):
        data[i] = data[i].split("\t")
    for i in range(len(data) - 1):
        X[i] = data[i][1]
        y[i] = data[i][0]
    # print(X)
    # print(y)
    return X, y, data


"""
  cleanData()
    Parameters: X
      X => original feature data (array/list of strings)

    this function cleans the feature data. It removes punctuation,
    converts letters to lower case, creates a bag of words vector,
    filters out stop words, includes 1-grams and 2-grams, and normalizes
    the frequences based on document length.
"""


def cleanData(X):
    # create the transform
    countVect = CountVectorizer()

    # use regular expression to convert text to tokens
    tokenizer = TreebankWordTokenizer()
    countVect.set_params(tokenizer=tokenizer.tokenize)

    # remove english stop words
    countVect.set_params(stop_words="english")

    # include 1-grams and 2-grams
    countVect.set_params(ngram_range=(1, 2))

    # transform text to bag of words vector using parameters
    X_counts = countVect.fit_transform(X)

    # normalize counts based on document length
    # weight common words less (is, a, an, the)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    return X_tfidf


"""
  naiveBayesClassifier()
      Parameters: Xtfidf, labels, description
      Xtfidf => tfidf features
      labels => labels of the data (array)
      description => string 

      Returns: classifier (naive bayes classifier)

      This is the naive bayes classifier function
      calls 3-fold cross validation for accuracy and f1 macro score.
"""


def naiveBayesClassifier(Xtfidf, labels, description):
    # train a naive Bayes classifier on data, nultinomial for discrete features

    # special classifier when we want to weight the sample data
    if description == "Naive Bayes classifier 3-fold cross validation on WEIGHTED data":
        weight, sampleWeights = weights(Xtfidf, labels)
        classifier = MultinomialNB().fit(Xtfidf, labels, sample_weight=sampleWeights)

    # whenever we do not want to weight the sample data
    else:
        classifier = MultinomialNB().fit(Xtfidf, labels)

    print("classifier: ", classifier)
    print(description)

    # get the scores from 3-fold cross validation
    accuracyScores = cross_val_score(
        classifier, Xtfidf, labels, cv=3, scoring="accuracy"
    )
    print("accuracies: ", accuracyScores)

    # get f1 macro score
    f1MacroScores = cross_val_score(
        classifier, Xtfidf, labels, cv=3, scoring="f1_macro"
    )
    print("f1 macro: ", f1MacroScores)

    print(
        "Number of SMS messages: ",
        len(labels),
        "\n 3-fold accuracy: ",
        np.mean(accuracyScores),
    )
    print(" 3-fold f1 macro score: ", np.mean(f1MacroScores))

    print()
    return classifier


"""
  detailedReport()
    Parameters: Xtfidf, labels, classifier
    Xtfidf => tfidf features
    labels => labels of the data (array)
    classifer => type of classifier model

    This function does not use 3-fold cross validation and does simple
    train_test_split on the data and uses the naive Bayes classifier. Prints
    confusion matrix, accuracy, and f1 macro score.
"""


def detailedReport(Xtfidf, labels, classifier):
    # non 3-fold cross validation
    print("Not using 3-fold cross validation, simple train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(Xtfidf, labels, test_size=0.33)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # get unique labels
    uniqueLabels = collections.Counter(labels)
    uniqueLabels = list(uniqueLabels.keys())

    # confusion matrix,
    print("Confusion matrix:")
    pprint.pprint(uniqueLabels, width=200)
    print(confusion_matrix(y_test, y_pred))

    accuracyScore = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracyScore)
    f1Macro = f1_score(y_test, y_pred, average="macro")
    print("F1 Macro Score:", f1Macro)
    print()


"""
  plotDataBar()
    Parameters: X, y
      X => tfidf features
      y => labels

    This function plots graphs for the different sample sizes after they were
    resampled. Through oversampling, undersampling, and weights. To show that they 
    were done properly.

"""


def plotDataBar(X, y):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax_arr = (ax1, ax2, ax3, ax4)

    originalLabels = collections.Counter(y)
    uniqueLabels = list(originalLabels.keys())

    # original data plot
    X_original = X
    y_original = y
    originalLabels = collections.Counter(y_original)
    OGvalues = list(originalLabels.values())
    ax1.bar(uniqueLabels, OGvalues, color="blue")
    ax1.set_title("Original Data set")
    ax1.set(ylabel="# of occurrences", xlabel="Labels")

    # undersample plot
    X_under, y_under = undersample(X, y)
    underLabels = collections.Counter(y_under)
    underValues = list(underLabels.values())
    ax2.bar(uniqueLabels, underValues, color="red")
    ax2.set_title("Undersample majority class")
    ax2.set(ylabel="# of occurrences", xlabel="Labels")

    # oversample plot
    X_over, y_over = oversample(X, y)
    overLabels = collections.Counter(y_over)
    overValues = list(overLabels.values())
    ax3.bar(uniqueLabels, overValues, color="green")
    ax3.set_title("Oversample minority class")
    ax3.set(ylabel="# of occurrences", xlabel="Labels")

    # weighted plot
    weight, weightedArray = weights(X, y)
    weightedLabels = collections.Counter(y_original)
    weightedValues = list(weightedLabels.values())
    for index, value in enumerate(weightedValues):
        weightedValues[index] = value * weight[index]

    # print("Weights: ", weight)
    ax4.bar(uniqueLabels, weightedValues, color="yellow")
    ax4.set_title("Weighted data")
    ax4.set(ylabel="Weight * (# of occurrences)", xlabel="Labels")


"""
  oversample()
    Parameters: X, y
      X => tfidf features
      y => labels
    Returns: adjusted X, y based on oversampling

    This function uses the SMOTE function to oversample the minority class
"""


def oversample(X, y):
    # define an oversampling method
    oversample = SMOTE()

    # resample data
    X, y = oversample.fit_sample(X, y)

    return X, y


"""
  undersample()
    Parameters: X, y
      X => tfidf features
      y => labels
    Returns: adjusted X, y based on undersampling

    This function uses the NearMiss function to undersample the majority class
"""


def undersample(X, y):
    # define an undersampling method
    undersample = NearMiss(version=1)

    # resample data
    X, y = undersample.fit_sample(X, y)

    return X, y


"""
  weights()
    Parameters: X, y
      X => tfidf features
      y => labels
    Returns: returns 2D weights array to be able to plot, and array length of y that
      has weights in corresponding indexes of labels

    This function calculates the weights of the corresponding labels to be balanced
"""


def weights(X, y):
    weights = []
    weightArray = np.empty(len(y))
    total = 0
    weightedLabels = collections.Counter(y)
    weightedValues = list(weightedLabels.values())
    # get total data points
    for value in weightedValues:
        total += value

    # find the weights of each label
    for value in weightedValues:
        weights.append(total / (2 * value))

    # # make n_sampls array of weights corresponding to correct label
    # for index, label in enumerate(y):
    #   if label == "ham":
    #       weightArray[index] = weights[0]
    #   else:
    #     weightArray[index] = weights[1]

    # print("Labels", y)
    # print("Weighted array", weightArray)

    # make n_sampls array of weights corresponding to correct label
    sampleWeights = class_weight.compute_sample_weight("balanced", y)

    return weights, sampleWeights


if __name__ == "__main__":
    X, y, data = getData()
    X_tfidf = cleanData(X)

    # original data set 3-fold cross validation
    description = "Naive Bayes classifier 3-fold cross validation on ORIGINAL data"
    naiveClassifier = naiveBayesClassifier(X_tfidf, y, description)

    # undersampling the majority
    description = "Naive Bayes classifier 3-fold cross validation on UNDERSAMPLED data with NearMiss"
    X_under, y_under = undersample(X_tfidf, y)
    overNaiveBayesClassifier = naiveBayesClassifier(X_under, y_under, description)

    # oversampling the minority
    description = (
        "Naive Bayes classifier 3-fold cross validation on OVERSAMPLED data with SMOTE"
    )
    X_over, y_over = oversample(X_tfidf, y)
    underNaiveBayesClassifier = naiveBayesClassifier(X_over, y_over, description)

    # Weighting the data points based on class imbalance.
    description = "Naive Bayes classifier 3-fold cross validation on WEIGHTED data"
    naiveClassifier = naiveBayesClassifier(X_tfidf, y, description)

    # Results without 3-fold cross validation and plots of data
    detailedReport(X_tfidf, y, naiveClassifier)
    plotDataBar(X_tfidf, y)
