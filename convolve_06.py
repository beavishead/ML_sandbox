import numpy as np

def convolve(mtx,wind,n,m,k):

    """
    Apply the convolution to the given matrix (mtx) with the sliding window (wind)
    Each element of the output matrix is a weighted sum of a k*k block from the input
    matrix with weights given by the convolution kernel

    Args:
        mtx (list : N*M): the inpiut matrix of the dimensions N*M.
        wind (list : k*k): the sliding window of dimensions k*k
        n - N
        m - M
        k - k

    Returns:
        conv: the matrix of dimensions (n-k+1)*(m-k+1).
    """
    mtx = np.array(mtx)
    wind = np.array(wind)
    conv = np.ones((n-k+1,m-k+1))
    for i in range(n-k+1):
        for ii in range(m-k+1):
            conv[i][ii] = np.sum(np.multiply(mtx[i:i+k,ii:ii+k], wind))
    return conv
