import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, cheby1, lfilter
from numpy.polynomial import polynomial as poly


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
    grad = 3
    p = np.random.randint(-10,10, size=grad+1)
    q = np.random.randint(-10,10, size=grad+1)
    # print(p,q)
    result = poly.polymul(p, q)

    p = np.fft.fft(p)
    q = np.fft.fft(q)
    r = signal.convolve(p, q)
    r = np.fft.ifft(r).real

    print(r,"\n\nThe correct result: ",result)
    # idk nu merge


def fereastra_drept(N):
    return np.ones(N)

def fereastra_hanning(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))

def semnal_sin(n):
    f = 100
    A = 1
    k = 0
    return A * np.sin(2 * np.pi * f * n * k)

def ex3():
    N=200
    n = np.linspace(0, 1, N)

    # Primul plot
    plt.subplot(2, 1, 1)
    plt.plot(n, semnal_sin(fereastra_drept(N)))
    plt.title("Dreptunghiulara")
    plt.subplot(2, 1, 2)
    plt.plot(n, semnal_sin(fereastra_hanning(N)))
    plt.title("Hanning")
    plt.show()

    # Al doilea plot
    plt.subplot(2, 1, 1)
    plt.plot(n, semnal_sin(n)*fereastra_drept(N))
    plt.title("Dreptunghiulara")
    plt.subplot(2, 1, 2)
    plt.plot(n, semnal_sin(n)*fereastra_hanning(N))
    plt.title("Hanning")
    plt.show()


def ex4():
    data = np.genfromtxt("Train.csv", delimiter=',', names=True, dtype=None, encoding=None)
    x = data['Count']
    x = x[:3 * 24]
    print(x)
    w_sizes = [5, 9, 13, 17]

    for w in w_sizes:
        filtered_signal = np.convolve(x, np.ones(w) / w, 'valid')
        plt.plot(filtered_signal, label=f'w = {w}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Exercitiul 1.
    # Operatia de convolutie repetata a lui x la sine formeaza o gausiana

    # ex1()
    # ex2()
    # ex3()
    ex4()
