import numpy as np

def convolve(mtx,wind,n,m,k):

    """
    Apply the convolution to the given matrix (mtx) with the sliding window (wind)
    At every step the convolution window is fit to the matrix , all the corresponding
    elements get multiplied and further summed over between each other

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
