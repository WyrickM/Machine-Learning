########################################################################
# Mantz Wyrick
# Machine Learning Homework #3.2
#
# K-means clustering
#
########################################################################

from google.colab import drive

drive.mount("/content/gdrive")

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import collections


def euclidean_distance(x, y):
    distance = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2
    return math.sqrt(distance)


"""
  update_centers function
    Arguments: 
      data => numpy array of pixel data that we get from the image
      labels => numpy array of labels of each data point that corresponds to closest cluster center
      centers => numpy array of centers of each cluster

    Returns:
      updated centers by taking the mean of the labels of each cluster
"""


def update_centers(data, labels, centers):
    # compute the new cluster centers as means of the cluster members

    # these print statements helped me understand what all of the data looked like
    # print("labels", labels)
    # print("length of labels", len(labels))
    # labels = np.array(labels)
    # print("Count of labels that equal 0", np.count_nonzero(labels == 0))
    # print("Count of labels that equal 1", np.count_nonzero(labels == 1))
    # print("Count of labels that equal 2", np.count_nonzero(labels == 2))

    for index, center in enumerate(centers):  # iterate through the centers
        cluster = np.zeros(
            (np.count_nonzero(labels == index), 3), dtype=int
        )  # create an empty array to be able to get mean of clusters
        cluster_index = 0
        # print("label 0", labels[0][0])
        for i, element in enumerate(
            labels
        ):  # put all data points with same label in one cluster
            if element[0] == index:
                cluster[cluster_index] = data[i]  # putting data points in one cluster
                cluster_index += 1
        centers[index] = np.mean(
            cluster, axis=0, dtype=int
        )  # taking the mean of the cluster, axis=0 means take the means of the columns

    return centers


def kmeans(data, k, max_iterations):
    random.seed()
    n = len(data)
    centers = np.zeros((k, 3))
    for i in range(k):  # pick random points for initial centers
        r = random.randrange(0, n - 1, 1)
        centers[i] = data[r]
    labels = np.zeros((len(data), 1), dtype=int)  # initial cluster memberships
    # update the clusters until max_iterations is reached
    # TO DO
    iteration = 0
    for iteration in range(max_iterations):
        # print("Iteration %i out of %i" %(iteration, max_iterations))
        for i in range(n):
            distances = np.zeros(k)  # k is number of clusters
            for j in range(
                k
            ):  # for each cluster center label point that to closest center
                distances[j] = euclidean_distance(
                    data[i], centers[j]
                )  # get the distance from the specific point to the current center
            labels[i] = [
                min(range(len(distances)), key=distances.__getitem__)
            ]  # take the minimum distance from point to all centers, and label it as/with the closest center
        centers = update_centers(data, labels, centers)  # update the centers

    return labels, centers


def main():
    image = cv2.imread(
        "/content/gdrive/My Drive/WSU/CPTS/CPTS 437/Colab Notebooks/HW3/image.png"
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # map onto our color space

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))  # Here -1 means unknown dimension
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    k = 3
    max_iterations = 2
    labels, (centers) = kmeans(pixel_values, k, max_iterations)
    # you can uncomment this next line to see how it could look after kmeans
    # _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # uncomment this line to see scikit learn kmeans.

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image)
    plt.show()


main()
