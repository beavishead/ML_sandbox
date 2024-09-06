"""
TASK: count the "cats" (the distinct shapes) on the 2d matrix. Every shape is surrounded
by the "background" zeros.

ALGORITHMS AND TECHNIQUES: deque, padding, movements' matrix, breadth-first search
"""

import sys
import numpy as np
from collections import deque

#input
cat_matrix=[]

#padding
for line in sys.stdin.readlines():
    cat_matrix.append(list(map(int,line.strip('\n').split())))
cat_matrix = np.pad(cat_matrix,pad_width=1)

#create mask-matrix
dims = cat_matrix.shape
mask_matrix= np.zeros((dims),dtype=int)

## implement breadth-first algorythm ##

#function that returns the elements around coordinate (x,y)
def check_the_surroundings(x_coord,y_coord):
    movements_i = [-1,1,0,0]
    movements_j = [0,0,-1,1]
    movements =[]
    for _ in range(4):
        movements.append((x_coord+movements_i[_],y_coord+movements_j[_]))
    return movements

cnt=0
for i in range(len(cat_matrix)):
    for j in range(len(cat_matrix[i])):
        if cat_matrix[i][j]==1 and mask_matrix[i][j]==0:
            cnt+=1
            mask_matrix[i][j] = cnt
            d = deque([(i,j)])
            while d:
                x_crd, y_crd = d.popleft()
                for x,y in check_the_surroundings(x_crd,y_crd):
                    if cat_matrix[x][y]==1 and mask_matrix[x][y]==0:
                        d.append((x,y))
                        mask_matrix[x][y] = cnt
print(cnt)
for i in mask_matrix[1:-1]:
    print(" ".join(list(map(str,i[1:-1]))))


