import numpy as np
import matplotlib.pyplot as plt


def ex1():
    N = 8
    n_arr = np.linspace(0, N-1, N)

    arr_cos = np.zeros((N, N), dtype=complex)
    arr_sin = np.zeros((N, N), dtype=complex)

    for m in n_arr:
        for k in n_arr:
            arr_cos[int(m), int(k)] = np.cos(2 * np.pi * m * k / N)
            arr_sin[int(m), int(k)] = 1j * np.sin(2 * np.pi * m * k / N)

    fig, axs = plt.subplots(8, 2, figsize=(10, 16))

    for m in n_arr:
        axs[int(m), 0].plot(arr_cos[int(m), :].real)
        axs[int(m), 1].plot(arr_sin[int(m), :].imag)
    plt.show()

    F = np.zeros((N, N), dtype=np.complex64)
    for i in range(N):
        for j in range(N):
            F[i, j] = np.exp(-2j * np.pi * i * j / N)
    f_conj_transpose = np.conj(F).T
    product = np.dot(f_conj_transpose, F)

    is_unitary = np.allclose(product, N * np.eye(N))

    print("\nEste matricea Fourier unitara?", is_unitary)


def semnal_sin(t):
    return np.sin(2 * 800 * np.pi * t)


def ex2():
    # Figura 1
    n = np.linspace(0, 1, 160)
    y = semnal_sin(n)*np.exp(-2j * np.pi * n)

    plt.plot(n, semnal_sin(n))
    plt.show()
    plt.plot(y.real, y.imag)
    plt.show()

    # Figura 2
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=2.0)

    z = semnal_sin(n)*np.exp(-2j * np.pi * n * 1)
    axs[0, 0].set_title("w=1")
    axs[0, 0].plot(z.real, z.imag)
    z = semnal_sin(n) * np.exp(-2j * np.pi * n * 2)
    axs[0, 1].set_title("w=2")
    axs[0, 1].plot(z.real, z.imag)
    z = semnal_sin(n) * np.exp(-2j * np.pi * n * 7)
    axs[1, 0].set_title("w=7")
    axs[1, 0].plot(z.real, z.imag)
    z = semnal_sin(n) * np.exp(-2j * np.pi * n * 800)
    axs[1, 1].set_title("w=800")
    axs[1, 1].plot(z.real, z.imag)
    plt.show()

def ex3():
    return 0


if __name__ == "__main__":
    ex1()
    ex2()
    ex3()
