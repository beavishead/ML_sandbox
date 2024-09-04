"""
The following snippet of code provides the metrics of similarity between two sets of
predictions A and B. The metrics is the share of pairs of objects for which A and B
predicted the same class devided by the total amount of pairs.

EXAMPLE:
    A: 1 0
    B: 1 0
OUTPUT:
    1/1
EXPLANATION:
    total number of pairs of objects = 1
    number of pairs for which A and B predict the same class for every object = 1
    the irreducable fraction : 1/1

EXAMPLE #2:
    A: 1 1
    B: 1 0
OUTPUT:
    0/1
"""

from collections import defaultdict
from fractions import Fraction

def count_differing_pairs(n, k, m):
    index_pairs_k = defaultdict(list)
    index_pairs_m = defaultdict(list)

    # Create dictionaries of indices for k and m
    for i in range(n):
        index_pairs_k[k[i]].append(i)
        index_pairs_m[m[i]].append(i)

    differing_pairs = 0

    # Count differing pairs based on k
    for indices in index_pairs_k.values():
        if len(indices) > 1:
            count = len(indices)
            total_pairs = count * (count - 1) // 2
            same_pairs = sum(
                1 for i in range(len(indices))
                for j in range(i + 1, len(indices))
                if m[indices[i]] == m[indices[j]]
            )
            differing_pairs += total_pairs - same_pairs

    # Count differing pairs based on m
    for indices in index_pairs_m.values():
        if len(indices) > 1:
            count = len(indices)
            total_pairs = count * (count - 1) // 2
            same_pairs = sum(
                1 for i in range(len(indices))
                for j in range(i + 1, len(indices))
                if k[indices[i]] == k[indices[j]]
            )
            differing_pairs += total_pairs - same_pairs

    return differing_pairs


def main():
    # Input
    n = int(input())
    k = list(map(int, input().split()))
    m = list(map(int, input().split()))

    # Calculate the total number of pairs
    total_pairs = n * (n - 1) // 2

    # Calculate the number of differing pairs
    differing_pairs = count_differing_pairs(n, k, m)

    # Calculate similarity
    numerator = total_pairs - differing_pairs
    denominator = total_pairs

    a = Fraction(numerator, denominator)

    # Output result
    if a == 0:
        print(f'0/{denominator}')
    elif a == 1:
        print('1/1')
    else:
        print(a)


# Execute main function
main()
