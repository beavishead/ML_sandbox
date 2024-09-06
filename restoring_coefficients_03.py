"""
THE FIND-COEF TASK ACCOMPLISHED BY USING THE GRADIENT DESCENT
"""

import sys
import numpy as np

# Reading data
inp = list(map(lambda x: x.split(), sys.stdin.readlines()))
x, y = zip(*inp)
x = np.array(list(map(float, x)))
y = np.array(list(map(float, y)))
n = float(len(inp))

# Initialize a, b, c
a, b, c = [np.random.ranf()]*3

# Initialize hyperparameters
alpha = 0.0001  # Smaller learning rate to stabilize training

for i in range(1000):
    # Defining function
    y_hat = (a * np.sin(x) + b * np.log(x))**2 + c * np.power(x, 2)

    # Computing the errors
    errors = y-y_hat

    # Counting the derivatives with respect to the cost function
    da = np.mean(4 * errors * (a * np.sin(x) + b * np.log(x)) * np.sin(x))
    db = np.mean(4 * errors * (a * np.sin(x) + b * np.log(x)) * np.log(x))
    dc = np.mean(2 * errors * np.power(x, 2))

    # Update rule
    a += alpha * da
    b += alpha * db
    c += alpha * dc

    # Printing the cost
    cost = np.mean(np.sqrt(np.power(errors, 2)))
    print(da,db,dc)
    print("Cost after iter", i, ":", cost)

print('%.2f'%a,'%.2f' % b,'%.2f' % c)
