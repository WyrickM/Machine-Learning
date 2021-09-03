########################################################################
# Mantz Wyrick
# Machine Learning Homework #3.1
#
# Decision tree learning
#
# Assumes discrete features. Examples may be inconsistent. Stopping condition for tree
# generation is when all examples have the same class, or there are no more features
# to split on (in which case, use the majority class). If a split yields no examples
# for a particular feature value, then the classification is based on the parent's
# majority class.
#
########################################################################

import math
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount("/content/gdrive")


class TreeNode:
    def __init__(self, majClass):
        self.split_feature = -1  # -1 indicates leaf node
        self.children = {}  # dictionary of {feature_value: child_tree_node}
        self.majority_class = majClass


def build_tree(examples):
    if not examples:
        return None
    features = {}
    # collect sets of values for each feature index, based on the examples
    for feature_index in range(len(examples[0]) - 1):
        features[feature_index] = set([example[feature_index] for example in examples])
        # print("feature:?",features[feature_index]) # just seeing what prof means by features list
    return build_tree_1(examples, features)


def build_tree_1(examples, features):
    tree_node = TreeNode(majority_class(examples))
    # if examples all have same class, then return leaf node predicting this class
    if same_class(examples):
        return tree_node
    # if no more features to split on, then return leaf node predicting majority class
    if not features:
        return tree_node
    # split on best feature and recursively generate children
    best_feature_index = best_feature(features, examples)
    # print("best feature index", best_feature_index)
    tree_node.split_feature = best_feature_index
    # print("tree node", tree_node.split_feature)
    remaining_features = features.copy()
    remaining_features.pop(best_feature_index)
    # print("reminaing features:", remaining_features)
    for feature_value in features[best_feature_index]:
        split_examples = filter_examples(examples, best_feature_index, feature_value)
        # print("split examples", split_examples)
        tree_node.children[feature_value] = build_tree_1(
            split_examples, remaining_features
        )
    return tree_node


def majority_class(examples):
    # print("length of examples:", len(examples))
    classes = [example[-1] for example in examples]
    # print("length of classes:", len(classes))
    return max(set(classes), key=classes.count)


def same_class(examples):
    classes = [example[-1] for example in examples]
    return len(set(classes)) == 1


def best_feature(features, examples):
    # Return index of feature with lowest entropy after split
    best_feature_index = -1
    best_entropy = 2.0  # max entropy = 1.0
    for feature_index in features:
        se = split_entropy(feature_index, features, examples)
        # print("se: ", se)
        if se < best_entropy:
            best_entropy = se
            best_feature_index = feature_index
    return best_feature_index


def split_entropy(feature_index, features, examples):
    # Return weighted sum of entropy of each subset of examples by feature value.
    se = 0.0
    for feature_value in features[feature_index]:
        split_examples = filter_examples(examples, feature_index, feature_value)
        se += (float(len(split_examples)) / float(len(examples))) * entropy(
            split_examples
        )
    return se


def entropy(examples):
    classes = [example[-1] for example in examples]
    classes_set = set(classes)
    # print("classes set", classes_set)
    class_counts = [classes.count(c) for c in classes_set]
    # print("classes counts", class_counts)
    e = 0.0
    class_sum = sum(class_counts)
    for class_count in class_counts:
        if class_count > 0:
            class_frac = float(class_count) / float(class_sum)
            e += (-1.0) * class_frac * math.log(class_frac, 2.0)
    return e


def filter_examples(examples, feature_index, feature_value):
    # Return subset of examples with given value for given feature index.
    # return list(filter(lambda example: example[feature_index] == feature_value, examples))
    return list(
        filter(lambda example: example[feature_index] == feature_value, examples)
    )


def print_tree(tree_node, feature_names, depth=1):
    indent_space = depth * "  "
    if tree_node.split_feature == -1:  # leaf node
        print(
            indent_space + str(feature_names[-1]) + ": " + str(tree_node.majority_class)
        )
    else:
        for feature_value in tree_node.children:
            # print("feature names", feature_names)
            # print("tree node", tree_node.split_feature)
            print(
                indent_space
                + str(feature_names[tree_node.split_feature])
                + " == "
                + str(feature_value)
            )
            child_node = tree_node.children[feature_value]
            if child_node:
                print_tree(child_node, feature_names, depth + 1)
            else:
                # no child node for this value, so use majority class of parent (tree_node)
                print(
                    indent_space
                    + "  "
                    + str(feature_names[-1])
                    + ": "
                    + str(tree_node.majority_class)
                )


def classify(tree_node, instance):
    if tree_node.split_feature == -1:
        return tree_node.majority_class
    child_node = tree_node.children[instance[tree_node.split_feature]]
    if child_node:
        return classify(child_node, instance)
    else:
        return tree_node.majority_class


def read_data():
    """
    Reading the data that was uploaded.
    """
    data = np.loadtxt(
        fname="/content/gdrive/My Drive/WSU/CPTS/CPTS 437/Colab Notebooks/HW1/alldata.csv",
        delimiter=",",
    )
    return data


def eq_freq_binning(training_data):
    """
    Equal freqency binning function. Calls sort data on all of the data, binning the
    entire data set into x number of bins.

    For homework purposes only spliting into two bins.
      In future try to add more bins
    return: list of all the bins from the data
    """
    # setting the number of bins to 4, can be any number, too many bins results in overfitting
    number_bins = 2
    bin_values = []  # return item: list
    copy_data = np.array(training_data)
    # print(all_data)
    for column in copy_data.T[:-1]:
        sorted_column = sorted(column)
        halfway_index = len(sorted_column) / 2  # index of the element in the middle
        halfway_value = sorted_column[
            halfway_index
        ]  # value of the in the middle of the column
        bin_values.append(halfway_value)  # add new element to end of list
    return bin_values


def get_train_test_data(all_data):
    """
    This function splits all of the data into training data and testing data.
    Training data: 80% of total data
    Testing data: 20% of total data
    return: training data list, testing data list
    """
    train = []
    test = []
    all_data = list(all_data)
    size_of_data = len(all_data)
    # get the total amount of data entries for the test data
    size_of_test = int(
        len(all_data) * 0.2
    )  # set the size of test data to be 20% of all data
    for count in range(0, size_of_test):
        size_of_data = len(all_data)
        test.append(
            all_data.pop(random.randint(0, size_of_data - 1))
        )  # randomly pick data entries
    train = all_data  # set new all_data (missing all test_data) to training data
    return train, test


def convert_data_based_on_bins(data, halfway_values):
    """
    This function converts the passed in data set and converts
    the data of that set based on the corresponding halfway value

    In future for more than two bins. halfway_values will be a list of lists
    with be minimum intervals and will iterate through list comparing and setting
    feature based on index of halfway_values.
    For example: if column[row_index] = 2 and halfway_values = [[0,1,3,4]]
    then column[row_index] will be changed to halfway_values_index + 1 (2)

    returns: a copy of the data passed with entries either 2 or 3.
    """
    copy_data = np.array(data)
    for column_index, column in enumerate(copy_data.T[:-1]):
        for row_index in range(0, len(column)):
            if column[row_index] < halfway_values[column_index]:
                column[
                    row_index
                ] = 2  # python would not let me set column[row_index] to be a string, casting error. Do not know why. Would prefer strings and have them say bin1, bin2,...
            else:
                column[row_index] = 3
    return copy_data


if __name__ == "__main__":
    feature_names = ["Color", "Type", "Origin", "Stolen"]

    examples = [
        ["Red", "Sports", "Domestic", "Yes"],
        ["Red", "Sports", "Domestic", "No"],
        ["Red", "Sports", "Domestic", "Yes"],
        ["Yellow", "Sports", "Domestic", "No"],
        ["Yellow", "Sports", "Imported", "Yes"],
        ["Yellow", "SUV", "Imported", "No"],
        ["Yellow", "SUV", "Imported", "Yes"],
        ["Yellow", "SUV", "Domestic", "No"],
        ["Red", "SUV", "Imported", "No"],
        ["Red", "Sports", "Imported", "Yes"],
    ]
    tree = build_tree(examples)
    print("Tree:")
    print_tree(tree, feature_names)
    test_instance = ["Red", "SUV", "Domestic"]
    test_class = classify(tree, test_instance)
    print("\nTest instance: " + str(test_instance))
    print("  class = " + test_class)
    print("\n\n")

    x_plots = []
    y_coordinates = []
    count = 0

    # sorry for the hard coded test cases, will change to loop in future.

    # build the one tree to use for testing.
    data = read_data()
    training_data, test_data = get_train_test_data(data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    tree = build_tree(list(training_data))

    # test case 1, data size 100, training size 80:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:100, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    # only build one tree, use this tree for all tests
    # should build tree on all data then do these test.
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    features = []
    for index in range(1, len(training_data[0]) + 1):
        features.append("feature: " + str(index))
    features[-1] = "-1 or 1"
    # print_tree(tree, features)

    count = 0
    # test case 2, data size 200, training size 160:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:200, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 3, data size 300, training size 240:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:300, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 4, data size 400, training size 320:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:400, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 5, data size 500, training size 400:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:500, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 6, data size 600, training size 480:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:600, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 7, data size 700, training size 560:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:700, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 8, data size 800, training size 640:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:800, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 9, data size 900, training size 720:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:900, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    count = 0
    # test case 10, data size 1000, training size 800:
    data = read_data()
    random.shuffle(data)
    sample_data = data[:1000, :]
    training_data, test_data = get_train_test_data(sample_data)
    bin_values = eq_freq_binning(training_data)
    training_data = convert_data_based_on_bins(training_data, bin_values)
    test_data = convert_data_based_on_bins(test_data, bin_values)
    test_data = np.array(test_data)
    copy_test_data = test_data[:, :-1]
    #  print("training", len(training_data))
    #  print("testing", len(test_data))
    # tree = build_tree(list(training_data))
    test_labels = []
    for test_instance in copy_test_data:
        test_class = classify(tree, test_instance)
        test_labels.append(test_class)

    actual_test_results = test_data[:, -1]
    for index in range(0, len(test_data)):
        if float(test_labels[index]) == float(actual_test_results[index]):
            count += 1
    accuracy = float(count) / float(len(test_data))
    #  print("accuracy", accuracy)
    #  plot_points.append((len(training_data), accuracy))
    x_plots.append(len(training_data))
    y_coordinates.append(accuracy)

    plt.scatter(x_plots, y_coordinates)
    plt.plot(x_plots, y_coordinates)
    plt.xlabel("Sample Data")
    plt.ylabel("Test Accuracy")
    plt.title("10 Test Cases")
    plt.show()

    # loop to get 10 more test cases to add to plot.
    # putting total cases to 20
    for test_case in range(11, 21):
        count = 0
        data = read_data()
        random.shuffle(data)
        sample_data = data[: (100 * test_case), :]
        training_data, test_data = get_train_test_data(sample_data)
        bin_values = eq_freq_binning(training_data)
        training_data = convert_data_based_on_bins(training_data, bin_values)
        test_data = convert_data_based_on_bins(test_data, bin_values)
        test_data = np.array(test_data)
        copy_test_data = test_data[:, :-1]
        # tree = build_tree(list(training_data))
        test_labels = []
        for test_instance in copy_test_data:
            test_class = classify(tree, test_instance)
            test_labels.append(test_class)

        actual_test_results = test_data[:, -1]
        for index in range(0, len(test_data)):
            if float(test_labels[index]) == float(actual_test_results[index]):
                count += 1
        accuracy = float(count) / float(len(test_data))
        x_plots.append(len(training_data))
        y_coordinates.append(accuracy)

    plt.scatter(x_plots, y_coordinates)
    plt.plot(x_plots, y_coordinates)
    plt.xlabel("Sample Data")
    plt.ylabel("Test Accuracy")
    plt.title("20 Test Cases")
    plt.show()
