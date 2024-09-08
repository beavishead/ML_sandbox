# **Machine Learning and Algorithm Implementations**

Welcome to my collection of machine learning and algorithm implementations. This repository contains various Python scripts that solve specific tasks and demonstrate different algorithms. Each script is accompanied by a brief explanation and an example of usage.

## **Table of Contents**

- [Overview](#overview)
- [Files and Implementations](#files-and-implementations)
  - [1. BFS Deque Implementation](#bfs-deque-implementation)
  - [2. Binary Classification](#binary-classification)
  - [3. Compression Medal](#compression-medal)
  - [4. Convolve Operation](#convolve-operation)
  - [5. Dots Generator](#dots-generator)
  - [6. Find Cat](#find-cat)
  - [7. Image Denoiser](#image-denoiser)
  - [8. K-Means Clustering](#k-means-clustering)
  - [9. Levenshtein Distance](#levenshtein-distance)
  - [10. Restoring Coefficients](#restoring-coefficients)
  - [11. Reverse Engineer Convolution Kernel](#reverse-engineer-convolution-kernel)
  - [12. Tags Accuracy](#tags-accuracy)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## **Overview**

This repository showcases a variety of implementations, ranging from basic algorithmic tasks to more complex machine learning models. Each file is designed to be standalone, allowing for easy understanding and execution of the code.

---

## **Files and Implementations**

### **1. BFS Deque Implementation**

**Filename**: `bfs_deque_implementation_01.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

**Usage**:

# Example usage of the BFS implementation

# K-Means Clustering Example

In this example, we apply the K-Means clustering algorithm to a dataset.

## **Algorithm Overview**

The K-Means algorithm aims to partition $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean.

The objective function is:

$$
J = \sum_{i=1}^{k} \sum_{j=1}^{n} || x_j^{(i)} - \mu_i ||^2
$$

Where:
- $x_j^{(i)}$ is the $j$-th point in cluster $i$.
- $\mu_i$ is the centroid of cluster $i$.

## **Resulting Clusters**

![K-Means Clustering Result](images/k_means_clusters.png)

### **3. Restoring function coefficients**

The chunk of code that restotores the function coefficients, given the datapoints $x$,$y$. The basic search algoruthm - Gradient Descent

The formula of the given function:

$$
f(x) = \left( (a + \epsilon_a) \sin{x} + (b + \epsilon_b) \ln{x} \right)^2 + (c + \epsilon_c) x^2
$$
