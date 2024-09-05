"""
PROBLEM STATEMENT: KMEANS CLUSTERING

Given the fixed number of clusters (k =402), the task is to split the dots (n =17038)
provided as (x,y) coordinate pairs in the dataset, so that it minimizes the eucledian
distance between these dots and the cluster centroids.

OVERVIEW OF THE SOLUTION:
1) Data Reading: Import the necessary libraries and read the data points from a CSV file.
2) Clustering: Use the K-Means algorithm to cluster the data into k clusters.
3) Cost Calculation: Calculate the cost associated with each cluster based on the Euclidean distance of the points to their centroids.
4) Profit Calculation: Subtract the total cost from the given budget to determine the final profit.

BRIEF RECAP OF HOW THE KMEANS WORKS:
1)initial centroids randomly/assign to the random existing points
2)assign dots to their nearest centroids
3)recalculate the centroids as the mean values of the dots in the given cluster
4)repeat until convergence

"""

# Import libraries
import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

# Read datapoints
with open('static/data/kmeans_clustering/profit.txt') as data:
    reader = csv.reader(data)
    i = 0
    x_s = []
    y_s = []
    for row in reader:
        if i == 0:
            line = ''.join(row)
            k, n, cost, budget = list(map(int, line.split()))
            i += 1
        else:
            line = ''.join(row)
            x, y = list(map(int, line.split()))
            x_s.append(x)
            y_s.append(y)
data = list(zip(x_s, y_s))

# Make a KMeans class instance with k centroids and fit data
kmeans = KMeans(n_clusters=k, n_init='auto').fit(data)

# Function to calculate the cost for clusters
dots_with_labels = list(zip(data, kmeans.labels_))
# Create a dictionary where the key is a centroid number and value is pairs of dots belonging to it
clusters = {}
for i in range(n):
    label = dots_with_labels[i][1]
    single_data_point = dots_with_labels[i][0]
    if label not in clusters:
        clusters[label] = [single_data_point]
    else:
        clusters[label].append(single_data_point)

# For each centroid, calculate the costs according to the given formula
total_cost = 0
for i in range(k):
    centroid = kmeans.cluster_centers_[i]
    cost_for_current_cluster = 0
    for j in range(len(clusters[i])):
        eucl_dist = euclidean_distances(np.array(centroid).reshape(1, -1), np.array(clusters[i][j]).reshape(1, -1))
        cost_for_current_cluster += ((eucl_dist ** 0.25 + 1) * cost) / len(clusters[i])
    total_cost += cost_for_current_cluster

# Output the profit for all clusters
profit = budget - total_cost
print(profit)
