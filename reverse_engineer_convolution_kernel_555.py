"""
REVERSE-ENGINEER THE CONVOLUTION KERNEL

TASK : given the input matrix and the output convolution, find the kernel

SOLUTION: Since the convolution is inherently the linear operation, we can represent
the task as solving the system of linear equations: A * K = B,
where A - the flattened patches of the size K from the input matrix(helper_mtx in code)
K - kernel(TO BE FOUND)
B - flattened output matrix
Therefore, K = A*-1 * B

"""
import numpy as np

# read input
n,m, k = map(int,input().split())
matrix = []
for _ in range(n):
    matrix.append(list(map(int,input().split())))
cnvl = []
for _ in range(n-k+1):
    cnvl.append(list(map(int,input().split())))

# build A
helper_mtx = []
for i in range(k):
    for ii in range(k):
        helper_helper_mtx = np.array(matrix)[i:i+k,ii:ii+k]
        helper_mtx.append(helper_helper_mtx.reshape(-1))

# build B
helper_cnvl = np.array(cnvl)[:k,:k].reshape(-1)

# find K
find_cnvl = np.dot(np.linalg.inv(helper_mtx),helper_cnvl.reshape(-1))
find_cnvl = [int(np.around(_)) for _ in find_cnvl]
find_cnvl = np.reshape(find_cnvl, (k,k))

# output as str
for row in find_cnvl:
    print(''.join(str(row).strip('[').strip(']')))

