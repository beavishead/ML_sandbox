# **Machine Learning and Algorithm Implementations**

Welcome to my collection of machine learning and algorithm implementations. This repository contains various Python scripts that solve specific tasks and demonstrate different algorithms. Each script is accompanied by a brief explanation and an example of usage.

## **Table of Contents**

- [Overview](#overview)
- [Files and Implementations](#files-and-implementations)
  - [1. BFS Deque Implementation](#1-bfs-deque-implementation)
  - [2. Dots generating function](#2-dots-generating-function)
  - [3. Restoring function coefficients](#3-restoring-function-coefficients)
  - [4. Levenshtein_distance](#4-levenshtein-distance)
  - [5. Tags accuracy](#5-tags-accuracy)
  - [6. Easy convolution](#6-easy-convolution)
  - [7. Reverse-engineer the convolution kernel](#7-reverse-engineer-the-convolution-kernel)
  - [8. K_means_clustering algorythm](#8-kmeansclustering-algorythm)
  - [9. Compression without information loss](#9-compression-without-information-loss)
  - [10. Restore damaged images](#10-restore-damaged-images)
  - [11. Find cat](#11-find-cat)
  - [12. Binary classification example](#12-binary-classification-example)
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

### **2. Dots generating function**

**Filename**: `dots_generator_02.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

## **Resulting Clusters**

![K-Means Clustering Result](images/k_means_clusters.png)

### **3. Restoring function coefficients**
**Filename**: `restoring_coefficients_03.py`

The chunk of code that restores the function coefficients, given the datapoints $x$, $y$. The basic convergence search algorithm - Gradient Descent

The formula of the given function:

$$
f(x) = \left( (a + \epsilon_a) \sin{x} + (b + \epsilon_b) \ln{x} \right)^2 + (c + \epsilon_c) x^2
$$

Where:
- $\epsilon_i$ are the random variables that take values from the interval [-0.001, 0.001].

### **4. Levenshtein distance**

**Filename**: `Levenshtein_distance_04.py`

**Description**: Levenshtein distance/ editor's distance, or the number of single-symbol operations to convert \
a string into another string.

Returns the count of such operations.

The algorithm is performed with building up the matrix every element of which is defined as:

<img src="static/pictures_for_readme/lev_dist.png" width="60%">

Where:
- $m(a,b) = 0$d $1$ otherwise; 
- $min(a,b,c)$ returns minimal value of the arguments.

### **5. Tags accuracy**

**Filename**: `tags_accuracy_05.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

### **6. Easy convolution**

**Filename**: `convolve_06.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

### **7. Reverse-engineer the convolution kernel**

**Filename**: `reverse_engineer_convolution_kernel_07.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

### **8. K_means_clustering algorythm**

**Filename**: `K_means_clustering_08.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

### **9. Compression without information loss**

**Filename**: `compression_medal_09.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

### **10. Restore damaged images**

**Filename**: `image_denoiser_10.py`

**Description**: The clgass for image restoration. It implements the simple algorithm to recover 85% of omitting pixel values through picking the random non-zero value in its nearest surrounding.

Some of the examples of restored images:

<img src="static/pictures_for_readme/comparison_04.png" width="50%">
<img src="static/pictures_for_readme/comparison_05.png" width="50%">

### **11. Find cat**

**Filename**: `find_cat_11.py`

**Description**: This script implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.

### **12. Binary classification example**
t
**Filename**: `binary_classification_12.py`

**Description**: This scriptc implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.