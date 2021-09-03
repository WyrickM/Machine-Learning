########################################################################
# Mantz Wyrick
# Machine Learning Homework #2
#
# Compare 4 classifiers: decision tree, k nearest neighbors, majority classifier,
# my own KNN
#
########################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split

classifiers = [
    (DecisionTreeClassifier(criterion="entropy"), "Decision Tree"),
    (KNeighborsClassifier(n_neighbors=5), "5NN"),
    (DummyClassifier(strategy="most_frequent"), "Simple Majority"),
]

"""
    read the data with load_digits()
    flatten the images, then split the data and return corresponding data.
"""


def get_data():
    # get data
    digits = load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # split data into 2/3 training and 1/3 testing
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=float(2 / 3), shuffle=True
    )
    return X_train, X_test, y_train, y_test


"""
      Take the specified classifers that can use scikit-learn
      Find the average for all of them and print them.
"""


def scikit_learn_classifiers():
    average_dict = {"Decision Tree": 0, "5NN": 0, "Simple Majority": 0}
    for index in range(3):
        print("Trial #" + str(index + 1) + ":")
        X_train, X_test, y_train, y_test = get_data()
        for classifier_structure, name in classifiers:
            classifier_structure.fit(X_train, y_train)
            newlabels = classifier_structure.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, newlabels)
            average_dict[name] = average_dict[name] + accuracy
            print("Classifier:", name)
            # print("new labels", newlabels)
            print("Accuracy:", metrics.accuracy_score(y_test, newlabels))
        print("\n")
    for name in average_dict:
        average_dict[name] = float(average_dict[name]) / float(index + 1)
        print("Classifier:", name)
        print("Average Accuracy:", average_dict[name], "\n")


# My implementation of K-Nearest Neighbors starts here

"""
  Computing the distance using the Euclidean formula
  Arguments: row of test data, row of training data
  returns: euclidean distance between the two rows/data points
"""


def euclidean_distance(test, train):
    distance = 0.0
    for index in range(len(train) - 1):
        distance += pow((test[index] - train[index]), 2)
    return np.sqrt(distance)


"""
  Locating the k nearest neighbors using euclidean distance.
  Arguments: all training data, a single test data point, k (amount of neighbors)
  returns: list of k nearest neighbors to the test point
"""


def nearest_neighbors(train, test, k_neighbors):
    distances = list()
    for row_index, train_row in enumerate(
        X_train
    ):  # iterate through all training data to get distances to the specific testing data
        dist = euclidean_distance(
            test, train_row
        )  # get the distance between each training data and the specific testing data
        distances.append((row_index, dist))
    distances.sort(
        key=lambda tup: tup[1]
    )  # sorting distances list based on the second element of tuple which is the distance
    neighbors = list()
    for index in range(k_neighbors):
        neighbors.append(
            distances[index][0]
        )  # append the first k neighbors of the sorted distances list to be the "nearest" neighbors
    return neighbors


"""
  This is the k nearest neighbors predicting function.
  Arguments: training data, training data labels, testing data, amount of neighbors
  returns: prediction based on neighbors
"""


def predict_class(train_data, test_data, train_labels, k_neighbors):
    predictions = list()
    for test_row in test_data:
        class_labels = list()
        neighbors = nearest_neighbors(
            train_data, test_row, k_neighbors
        )  # getting the nearest k neighbors to the testing data
        for neighbor in neighbors:
            class_labels.append(
                train_labels[neighbor]
            )  # getting the class labels of the training data that correspond to neighbors
        prediction = max(
            set(class_labels), key=class_labels.count
        )  # getting the most frequent/occuring neighbor and making it the prediction
        predictions.append(
            prediction
        )  # put all predictions in a list to return to compare to actual test labels
    return predictions


if __name__ == "__main__":
    scikit_learn_classifiers()
    print("\n")

    # using the same data for each k = 1, 3, 5, 7, 9 trial to see good accuracy changes

    accuracy_dict = {1: 0, 3: 0, 5: 0, 7: 0, 9: 0}
    # need to run 3 times and get averages of all trials.
    for trial in range(1, 4):
        X_train, X_test, y_train, y_test = get_data()
        print("Trial #%i: " % (trial))
        for k_neighbors in range(1, 11, 2):
            predictions = predict_class(X_train, X_test, y_train, k_neighbors)
            accuracy = metrics.accuracy_score(y_test, predictions)
            accuracy_dict[k_neighbors] = accuracy_dict[k_neighbors] + accuracy
            # print("Predictions of %i-nearest neighbors: " %(k_neighbors), predictions)
            print("Accuracy of %i-nearest neighbors:" % (k_neighbors), accuracy)
        print()

    print()

    for k_neighbors in accuracy_dict:
        print(
            "Average Accurace of %i-nearest neighbors:" % (k_neighbors),
            accuracy_dict[k_neighbors] / 3,
        )

    print()
    print(
        "The value for k that I would choose is k=5. I would choose k=5" + "\n"
        "even though it does not have the best accuracy but because it will be the"
        + "\n"
        "least likely to cause overfitting or underfitting. With k=1 or even k=3" + "\n"
        "there is a high chance of overfitting the data. With k=7 or k=9 you can" + "\n"
        "already start to see underfitting by the fact that the accuracies of k=7"
        + "\n"
        "and k=9 are less than k=5."
    )
