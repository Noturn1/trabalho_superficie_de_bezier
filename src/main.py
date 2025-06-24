import numpy as np
import matplotlib.pyplot as plt
import math as mt
import sys

def c(m ,j):
    return mt.factorial(m) / (mt.factorial(j) * mt.factorial(m - j))

def bez(v, j, m):
    return c(m, j) * (v ** j) * (1 - v) ** (m - j)

def p(u, v, j, m, k, n):
    return bez(v, j, m) * bez(u, k, n)

def main():

    if len(sys.argv) > 1:
        try:
            i = int(sys.argv[1])
        except ValueError:
            raise ValueError("O argumento deve ser um número inteiro.")
    else:
        print("Nenhum argumento fornecido. Usando valor padrão de 10.")
        i = 10

    # pontos
    u = np.linspace(0, 1, i)
    v = np.linspace(0, 1, i)

    # matriz 4x4 3D
    control_point = np.array([
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        [[0, 1, 0], [1, 1, 0], [2, 1, 5], [3, 1, 0]],
        [[0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0]],
        [[0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0]]
    ])
    # criar matriz 4x4 3D vazia
    superficie = np.zeros((i, i, 3))

    m = control_point.shape[0] - 1
    n = control_point.shape[1] - 1

    for u_it in range(i):
        for v_it in range(i):
            for j in range(m+1):
                for k in range(n+1):
                    superficie[u_it, v_it] += control_point[j, k] * p(u[u_it], v[v_it], j, m, k, n)

    # Plotando a superfície
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plotar a superfície de Bézier
    superficie = superficie.reshape((superficie.shape[0], superficie.shape[1], 3))
    for row in superficie:
        ax.plot([p[0] for p in row], [p[1] for p in row], [p[2] for p in row], 'b')
    for col in superficie.transpose(1, 0, 2):
        ax.plot([p[0] for p in col], [p[1] for p in col], [p[2] for p in col], 'b')

    # Conectar os pontos de controle com linhas horizontais e verticais
    control_point = control_point.reshape((control_point.shape[0], control_point.shape[1], 3))
    for row in control_point:
        ax.plot([p[0] for p in row], [p[1] for p in row], [p[2] for p in row], 'ro-')
    for col in control_point.transpose(1, 0, 2):
        ax.plot([p[0] for p in col], [p[1] for p in col], [p[2] for p in col], 'ro-')

    # Configurar etiquetas y título
    ax.set_title("Superficie de Bézier")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostrar el gráfico
    plt.show()

if __name__ == "__main__":
    main()