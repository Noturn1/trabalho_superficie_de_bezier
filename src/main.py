import math as mt
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def generate_random_control_points(
    grid_size, spacing, height_scale, smoothness, falloff_strength
):
    # Gerar altura base XY
    x = np.linspace(0, (grid_size - 1) * spacing, grid_size)
    y = np.linspace(0, (grid_size - 1) * spacing, grid_size)
    xx, yy = np.meshgrid(x, y)

    low_res_size = int(np.floor(grid_size * (1 - smoothness))) + 1
    if low_res_size < 2:
        low_res_size = 2  # Deve ser pelo menos 2x2 para interpolar

    # Gerar mapa de alturas aleatórias pequeno e de baixa resoluçao
    low_res_zz = np.random.randn(low_res_size, low_res_size)

    # Aumentar escala para tamanho grid alvo usando interpolaçao suave (bicúbica)
    zoom_factor = grid_size / low_res_size
    zz = zoom(low_res_zz, zoom_factor, order=3) * height_scale

    # Criar a aplicar mapa do fallout
    # Coordenadas de -1 a 1 para calcular distância do centro
    x_coords = np.linspace(-1, 1, grid_size)
    y_coords = np.linspace(-1, 1, grid_size)
    gx, gy = np.meshgrid(x_coords, y_coords)

    # Calcular distância do centro para cada ponto
    distance_from_center = np.sqrt(gx**2 + gy**2)

    # Normalizar distância
    distance_from_center /= np.sqrt(
        2
    )  # Distância máxima de um quadrado de -1 a 1 é raiz de 2

    # Criar o mapa de falloff. (1 - distância) o torna 1 no centro e 0 nos cantos.
    falloff_map = (1 - distance_from_center) ** falloff_strength

    # Aplicar valores do fallof ao Z
    zz *= falloff_map

    # Juntar tudo
    control_points = np.stack([xx, yy, zz], axis=-1)

    return control_points


def c(m, j):
    return mt.factorial(m) / (mt.factorial(j) * mt.factorial(m - j))


def bez(v, j, m):
    return c(m, j) * (v**j) * (1 - v) ** (m - j)


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

    control_point = generate_random_control_points(
        grid_size=6,  # Grid de pontos de controle
        spacing=1.0,  # Distancia entre pontos XY no plano
        height_scale=3.5,  # Altura das "montanhas"
        smoothness=0.6,  # 0.0 = spiky, 1.0 = very smooth
        falloff_strength=2,
    )

    # criar matriz 4x4 3D vazia
    superficie = np.zeros((i, i, 3))

    m = control_point.shape[0] - 1
    n = control_point.shape[1] - 1

    for u_it in range(i):
        for v_it in range(i):
            for j in range(m + 1):
                for k in range(n + 1):
                    superficie[u_it, v_it] += control_point[j, k] * p(
                        u[u_it], v[v_it], j, m, k, n
                    )

    # Plotando a superfície
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plotar a superfície de Bézier
    superficie = superficie.reshape((superficie.shape[0], superficie.shape[1], 3))
    for row in superficie:
        ax.plot([p[0] for p in row], [p[1] for p in row], [p[2] for p in row], "b")
    for col in superficie.transpose(1, 0, 2):
        ax.plot([p[0] for p in col], [p[1] for p in col], [p[2] for p in col], "b")

    # Conectar os pontos de controle com linhas horizontais e verticais
    control_point = control_point.reshape(
        (control_point.shape[0], control_point.shape[1], 3)
    )

    for row in control_point:
        ax.plot([p[0] for p in row], [p[1] for p in row], [p[2] for p in row], "ro-")
    for col in control_point.transpose(1, 0, 2):
        ax.plot([p[0] for p in col], [p[1] for p in col], [p[2] for p in col], "ro-")

    # Configurar etiquetas e título
    ax.set_title("Superficie de Bézier")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Definir visao de camera
    ax.view_init(elev=60, azim=90)

    # Plotar gráfico
    plt.show()


if __name__ == "__main__":
    main()

