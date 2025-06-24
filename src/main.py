import numpy as np
import matplotlib.pyplot as plt
import math as mt

def c(m ,j):
    return mt.factorial(m) / (mt.factorial(j) * mt.factorial(m - j))

def bez(v, j, m):
    return c(m, j) * (v ** j) * (1 - v) ** (m - j)

def p(u, v, j, m, k, n):
    return bez(v, j, m) * bez(u, k, n)

def main():
    # intervalo
    i = 10

    #pontos
    u = np.linspace(0, 1, i)
    v = np.linspace(0, 1, i)

    # matriz 4x4 3D
    control_point = np.array([
        [[0, 0, 0], [1, 2, 0], [2, 0, 0], [3, 1, 0]],
        [[0, 1, 1], [1, 3, 1], [2, 1, 1], [3, 2, 1]],
        [[0, 2, 2], [1, 4, 2], [2, 2, 2], [3, 3, 2]],
        [[0, 3, 3], [1, 5, 3], [2, 3, 3], [3, 4, 3]]
    ])
    # criar matriz 4x4 3D vazia
    superficie = np.zeros((i, i, 3))

    print(superficie)

    m = control_point.shape[0]
    n = control_point.shape[1]

    for u_it in range(i):
        for v_it in range(i):
            for j in range(m):
                for k in range(n):
                    superficie[u_it, v_it] += control_point[j, k] * p(u[u_it], v[v_it], j, m, k, n)


    # Plotando a superfície
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(superficie[:, :, 0], superficie[:, :, 1], superficie[:, :, 2], cmap='viridis')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('Superfície Bézier')
    plt.show()

if __name__ == "__main__":
    main()