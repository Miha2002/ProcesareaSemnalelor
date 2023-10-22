import numpy as np
import matplotlib.pyplot as plt

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t):
    return np.cos(280 * np.pi * t + np.pi / 3)

def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

def punctul_a():
    k = np.linspace(0, 0.03, 600)
    fig, axs = plt.subplots(1)
    fig.suptitle("A)")
    for i in k:
        axs.plot(i, 1, ".")
    plt.show()

def punctul_b():
    k = np.linspace(0, 0.03, 600)

    fig, axs = plt.subplots(3)
    fig.suptitle("B)")
    for i in k:
        axs[0].plot(i, x(i), ".")
        axs[1].plot(i, y(i), ".")
        axs[2].plot(i, z(i), ".")
    plt.show()

def punctul_c():
    k = np.linspace(0, 0.03, 6)

    fig, axs = plt.subplots(3)
    fig.suptitle("C)")
    for i in k:
        axs[0].plot(i, x(i), ".")
        axs[1].plot(i, y(i), ".")
        axs[2].plot(i, z(i), ".")
    plt.show()


if __name__ == "__main__":
    # Exercitiul 1
    punctul_a()
    punctul_b()
    punctul_c()

    # Exercitiul 2
