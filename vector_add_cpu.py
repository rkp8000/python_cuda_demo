import numpy as np
from timeit import default_timer as timer

def vector_add(a, b, c):
    for i in range(a.size):
        c[i] = a[i] + b[i]

def main():
    N = 32000000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)
    start = timer()
    vector_add(A, B, C)
    vector_add_time = timer() - start
    print('C[:5] = {}'.format(C[:5]))
    print('C[-5:] = {}'.format(C[-5:]))
    print('vector_add took {} seconds'.format(vector_add_time))

if __name__ == '__main__':
    main()

