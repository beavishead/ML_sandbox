"""
THE TASK : given two dots' generators and the 100 sets of dots(n=2000) predict which
of the generators were used to produce every set

THE SOLUTION: visualize, find the statistical measure that tells apart two distributions
(that turned out to be kurtosis). The following method helps to achieve accuracy >98%
"""

import math
import random as rd
import matplotlib.pyplot as plt
import numpy as np

def generate1():
    a = rd.uniform(0,1)
    b = rd.uniform(0,1)
    return (a * math.cos(2*math.pi*b), a*math.sin(2*math.pi*b))

def generate2():
    while True:
        x = rd.uniform(-1, 1)
        y = rd.uniform(-1, 1)
        if x ** 2 + y ** 2 > 1:
            continue
        return (x, y)
horiz_x = np.arange(1000)
dots =[]
dots2=[]
for _ in range(1000):
    xy_pair = generate1()
    x1y1_pair = generate2()
    dots.append(xy_pair)
    dots2.append(x1y1_pair)

#make a plot
fig, (ax1,ax2,ax3) = plt.subplots(3)
x,y = list(zip(*dots))
x1,y1 = list(zip(*dots2))
abs_y = [abs(_) for _ in y]
ax1.scatter(x,y,s=1.9)
ax2.scatter(x1,y1, s = 1.9,color='r')
ax3.hist(y,bins=50)
fig.set_size_inches(5,15)
ax1.set_xlim([-1.1,1.1])
ax1.set_ylim([-1.1,1.1])

plt.plot()
plt.show()

## Solution #1: through kurtosis
def kurtosis(ar):
    return (sum((ar - np.mean(ar)) ** 4) / len(ar)) / np.std(ar) ** 4

for i in range(100):
    pairs_dots = list(map(float, input().split()))
    x_coords = pairs_dots[0::2]
    if kurtosis(x_coords) >= 2.5:
        print("1")
    else:
        print("2")

