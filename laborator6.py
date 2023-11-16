import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def ex1():
    N = 100
    x = np.random.rand(N)

    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(x)
    x = signal.convolve(x, x)
    axs[0,1].plot(x)
    x = signal.convolve(x, x)
    axs[1,0].plot(x)
    x = signal.convolve(x, x)
    axs[1,1].plot(x)
    plt.show()


def ex2():
    grad = 20
    p = np.random.randint(100, size=grad+1)
    q = np.random.randint(100, size=grad+1)
    # print(p,q)
    r1 = signal.convolve(p, q)
    r2 = signal.fftconvolve(p, q)



if __name__ == "__main__":
    # Exercitiul 1.
    # Operatia de convolutie repetata a lui x la sine formeaza o gausiana

    # ex1()
    ex2()
