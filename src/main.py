import numpy as np
import matplotlib.pyplot as plt

def c(a ,b):
    return np.factorial(a) / (np.factorial(b) * np.factorial(a - b))

def bez(v, a, b):
    return c(a, b) * (v ** b) * (1 - v) ** (a - b)

def p(u, v, a, b, c, d):
    return bez(v, a, b) * bez(u, c, d)

def main():
    # matriz 4x4 3D
    control_point = np.array([
        [0, 0, 0], [1, 2, 0], [2, 0, 0], [3, 1, 0],
        [0, 1, 1], [1, 3, 1], [2, 1, 1], [3, 2, 1],
        [0, 2, 2], [1, 4, 2], [2, 2, 2], [3, 3, 2],
        [0, 3, 3], [1, 5, 3], [2, 3, 3], [3, 4, 3]
    ])

    superficie = np.array([[0, 0, 0] for _ in range(16)])

if __name__ == "__main__":
    main()