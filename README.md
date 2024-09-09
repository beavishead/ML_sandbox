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
I called it a SANDBOX , because it's my space where I can play around and test new approaches.
---

## **Files and Implementations**

### **1. BFS Deque Implementation**

**Filename**: `bfs_deque_implementation_01.py`

**Description**: The script is for finding of all paths on the n*m grid from multiple cells to the target. The legal moves are those equal to the knights move in chess.
-1 is returned in case the destination cannot be reached from at least one cell.

## Key Features:
Grid Initialization: A grid of size n x m is created.  
Breadth-First Search (BFS): The BFS algorithm explores all possible paths from the start point to cover all cells to the targer, iteratively.  
Path Calculation: The script calculates the sum of all paths needed to visit the target cell.  

## Inputs-outputs:
Input: Grid size, start coordinates, target.  
Output: The total distance to cover all targets or -1 if a target is unreachable  

### **2. Dots generating function**

**Filename**: `dots_generator_02.py`

**Description**: The task is given 2000 sets of 100 dots on a 2d plan, predict its generating function.


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

<img src="static/pictures_for_readme/lev_dist.png" width="30%">

Where:
- $m(a,b)$ = $0$ and $1$ otherwise; 
- $min(a,b,c)$ returns minimal value of the arguments.

### **5. Tags accuracy**

**Filename**: `tags_accuracy_05.py`

**Description**: The custom metric to measure the similarity between to sets of labels.

It accounts for the number of pairs of tags that are both equal in both sets of predictions, relative to the whole number of pairs.

g$$
\(\text{Similarity}(p,v) = 2 \times \frac{1}{n \cdot (n-1)} \sum_{i=1}^{n} \sum_{j=1}^{i-1} I[p_i = p_j] = I[v_i = v_j]\)
$$

### **6. Easy convolution**

**Filename**: `convolve_06.py`

**Description**: Just a convolution performed in pure numpy.

### **7. Reverse-engineer the convolution kernel**

**Filename**: `reverse_engineer_convolution_kernel_07.py`

**Description**: Find the kernel

**Convolution Operation:**

$$
O(x, y) = \sum_{i} \sum_{j} K(i, j) \cdot I(x+i, y+j)
$$

where:
- \( O(x, y) \) is the output image pixel at position \( (x, y) \).
- \( K(i, j) \) are the elements of the kernel.
- \( I(x+i, y+j) \) are the pixels from the input image that overlap with the kernel.
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

Solve the system of equations to find the values of \( K(i, j) \).

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

**Filename**: `binary_classification_12.py`

**Description**: This scriptc implements a Breadth-First Search (BFS) using a deque for efficient traversal in graph-based problems.