from numba import vectorize
import numpy as np
from timeit import default_timer as timer

@vectorize(['float32(float32, float32)'], target='cuda')
def vector_add(a, b):
    return a + b

def main():
    N = 32000000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)
    start = timer()
    C = vector_add(A, B)
    vector_add_time = timer() - start
    print('C[:5] = {}'.format(C[:5]))
    print('C[-5:] = {}'.format(C[-5:]))
    print('vector_add took {} seconds'.format(vector_add_time))

if __name__ == '__main__':
    main()

