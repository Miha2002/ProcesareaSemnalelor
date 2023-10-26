import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy
import math


def semnal_sin(t):
    return np.sin(400 * np.pi * t)

def ex1():
    N=8
    N_arr = np.linspace(0,7,8)
    x = semnal_sin(N_arr)

    arr_cos = np.zeros((N,N), dtype=complex)
    arr_sin = np.zeros((N,N), dtype=complex)

    for m in N_arr:
        for k in N_arr:
            arr_cos[int(m), int(k)] = np.cos(2 * np.pi * m * k / N)
            arr_sin[int(m), int(k)] = 1j * np.sin(2 * np.pi * m * k / N)

    plt.scatter(x*arr_cos.real,x*arr_sin.imag)
    plt.show()


if __name__ == "__main__":
    ex1()

