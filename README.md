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

---

## **Overview**

This repository showcases a variety of implementations, ranging from basic algorithmic tasks to more complex machine learning models. Each file is designed to be standalone, allowing for easy understanding and execution of the code.

I called it a SANDBOX , because it's my space where I can play around and test new approaches.

---

## **Files and Implementations**

### **1. BFS Deque Implementation**

**Filename**: `bfs_deque_implementation_01.py`

**Description**: The script is for finding of all paths on the n*m grid from multiple cells to the target. The legal moves are those equal to the knights move in chess.
-1 is returned in case the destination cannot be reached from at least one cell.
<br><br>
### **2. Dots generating function**

**Filename**: `dots_generator_02.py`

**Description**: The task is given 2000 sets of 100 dots on a 2d plan, predict its generating function.
<br><br>
### **3. Restoring function coefficients**
**Filename**: `restoring_coefficients_03.py`

The chunk of code that restores the function coefficients, given the datapoints $x$, $y$. The basic convergence search algorithm - Gradient Descent

The formula of the given function:

$$
f(x) = \left( (a + \epsilon_a) \sin{x} + (b + \epsilon_b) \ln{x} \right)^2 + (c + \epsilon_c) x^2
$$

Where:
- $\epsilon_i$ are the random variables that take values from the interval [-0.001, 0.001].
<br><br>
### **4. Levenshtein distance**

**Filename**: `Levenshtein_distance_04.py`

**Description**: Levenshtein distance/ editor's distance, or the number of single-symbol operations to convert \
a string into another string.

Returns the count of such operations.

The algorithm is performed with building up the matrix.

Given strings $S_1$ and $S_2$, the distance $d(S_1, S_2)$ is defined as:

$d(S_1, S_2) = D(M, N)$

where $D(i, j)$ is defined as:

$$
D(i, j) =
\begin{cases} 
0 & \text{if } i = 0 \text{ and } j = 0 \\
i & \text{if } j = 0 \text{ and } i > 0 \\
j & \text{if } i = 0 \text{ and } j > 0 \\
\min \{
D(i, j-1) + 1, \\
D(i-1, j) + 1, \\
D(i-1, j-1) + m(S_1[i], S_2[j]) 
\} & \text{if } j > 0 \text{ and } i > 0
\end{cases}
$$

Where:
- $m(a,b)$ = $0$ and $1$ otherwise; 
- $min(a,b,c)$ returns minimal value of the arguments.
<br><br>
### **5. Tags accuracy**

**Filename**: `tags_accuracy_05.py`

**Description**: The custom metric to measure the similarity between to sets of labels.

It accounts for the number of pairs of tags that are both equal in both sets of predictions, relative to the whole number of pairs.

$$
Similarity(p, v) = 2 \times \frac{1}{n \cdot (n-1)} \sum_{i=1}^{n} \sum_{j=1}^{i-1} I[p_i = p_j] = I[v_i = v_j]
$$
<br><br>
### **6. Easy convolution**

**Filename**: `convolve_06.py`

**Description**: Just a convolution performed in pure numpy.
<br><br>
### **7. Reverse-engineer the convolution kernel**

**Filename**: `reverse_engineer_convolution_kernel_07.py`

**Description**: Find the kernel

**Convolution Operation:**

$$
O(x, y) = \sum_{i} \sum_{j} K(i, j) \cdot I(x+i, y+j)
$$

where:
- $O(x, y)$ is the output image pixel at position \( (x, y) \).
- $K(i, j)$ are the elements of the kernel.
- $I(x+i, y+j)$ are the pixels from the input image that overlap with the kernel.
- The summations are over the kernel dimensions.

**Finding the Kernel:**

Given the system of linear equations derived from the convolution operation:

$$
\begin{aligned}
O_1 &= K(0,0) \cdot I_1 + K(0,1) \cdot I_2 + \cdots + K(2,2) \cdot I_9 \\
O_2 &= K(0,0) \cdot I_2 + K(0,1) \cdot I_3 + \cdots + K(2,2) \cdot I_{10} \\
&\vdots \\
O_n &= K(0,0) \cdot I_n + K(0,1) \cdot I_{n+1} + \cdots + K(2,2) \cdot I_{n+8}
\end{aligned}
$$

Solve the system of equations to find the values of $K(i, j)$.

More explanation in the file `reverse_engineer_convolution_kernel_07.py`
<br><br>
### **8. K_means_clustering algorythm**

**Filename**: `K_means_clustering_08.py`

**Description**: The code solves the problem of maximizing the retailer's profit. The task is to scatter the pick points on a map, with retailer profit defined as:

$$
Profit = C - \sum_{i=0}^{q} \sum_{h} \frac{c_{\text{cost}} \times |k_i|^4}{d_{h,c_i} + 1}
$$

where:

- $k_i$ - the set of houses for which the delivery point $c_i$ is the closest  
- $d_{h,c} = (x_h - c_x)^2 + (y_h - c_y)^2$ - the Euclidean distance from house $h$ in district $k$ to the delivery point $c$ in the same district  
- $C$ - the expected profit from all open delivery points  
- $c_{\text{cost}}$ - the cost of opening one delivery point.

I implement the kmeans clustering algorithm to find the optimal distribution of pick points on the map 
<br><br>
### **9. Compression without information loss**

**Filename**: `compression_medal_09.py`

**Description**: Perform PCA feature reduction, so that the mean linear regression error doesn't drop below the certain benchmark on the cross-validation set of features 
<br><br>
### **10. Restore damaged images**

**Filename**: `image_denoiser_10.py`

**Description**: The class for image restoration. It implements the simple algorithm to recover 85% of omitting pixel values through picking the random non-zero value in its nearest surrounding.

Some of the examples of restored images:

<img src="static/pictures_for_readme/comparison_04.png" width="50%">
<img src="static/pictures_for_readme/comparison_05.png" width="50%">
<br><br>

### **11. Find cat**

**Filename**: `find_cat_11.py`

**Description**: The script is provided as an example of using deque-structure for effective implementation of the BFS on a chess-like grid.
<br><br>
### **12. Binary classification example**

**Filename**: `binary_classification_12.py`

**Description**: The example of the data-mining pipeline, with testing different models(logistic regressino, gradient boosting, random tree) for the classification task.