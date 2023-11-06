import numpy as np
import time
import matplotlib.pyplot as plt

def semnal_ex1(n):
    return np.sin(2 * 25 * np.pi * n)

def ex1():
    valN = (128, 256, 512, 1024, 2048, 4096, 8192)

    for N in valN:
        n = np.linspace(0, N - 1, N)
        semnal = semnal_ex1(n)

        start = time.time() * 1000
        F = np.zeros((N, N), dtype=np.complex64)
        for i in range(N):
            for j in range(N):
                F[i, j] = np.exp(-2j * np.pi * i * j / N)

        F1 = np.dot(F, semnal)
        plt.plot(n, F1.real)
        plt.show()
        end1 = time.time() * 1000

        F2 = np.fft.fft(semnal_ex1(n),N)
        plt.plot(n, F2.real)
        plt.show()
        end2 = time.time() * 1000
        print("\nN=", N ,"\nOur version: ", round(end1-start), "ms\nNumpy version:", round(end2-end1),"ms")
    # N = 128
    # Our version: 2016ms
    # Numpy version: 393ms
    #
    # N = 256
    # Our version: 698ms
    # Numpy version: 335ms
    #
    # N = 512
    # Our version: 1918ms
    # Numpy version: 335ms
    #
    # N = 1024
    # Our version: 5372ms
    # Numpy version: 296ms
    #
    # N = 2048
    # Our version: 16299ms
    # Numpy version: 209ms
    #
    # N = 4096
    # Our version: 98032ms
    # Numpy version: 373ms
    #
    # N = 8192
    # Our version: 351055ms
    # Numpy version: 348ms


def semnal1_ex2(n):
    return 2 * np.cos(2 * 1 * np.pi * n)

def semnal2_ex2(n):
    return 3 * np.cos(2 * 5 * np.pi * n)

def semnal3_ex2(n):
    return 3 * np.cos(2 * 7 * np.pi * n)

def ex2():
    n = np.linspace(0, 1, 100)
    n_small = np.linspace(0, 1, 7)

    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=2.0)

    axs[0].plot(n, semnal1_ex2(n))
    axs[0].stem(n_small, semnal1_ex2(n_small), basefmt="None", linefmt="None", markerfmt="red")

    axs[1].plot(n, semnal2_ex2(n))
    axs[1].stem(n_small, semnal2_ex2(n_small), basefmt="None", linefmt="None", markerfmt="red")

    axs[2].plot(n, semnal3_ex2(n))
    axs[2].stem(n_small, semnal3_ex2(n_small), basefmt="None", linefmt="None", markerfmt="red")
    plt.show()

def ex3():
    n = np.linspace(0, 1, 200)
    n_small = np.linspace(0, 1, 15) # am schimbat frecventa din 7 -> 15

    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=2.0)

    axs[0].plot(n, semnal1_ex2(n))
    axs[0].stem(n_small, semnal1_ex2(n_small), basefmt="None", linefmt="None", markerfmt="red")

    axs[1].plot(n, semnal2_ex2(n))
    axs[1].stem(n_small, semnal2_ex2(n_small), basefmt="None", linefmt="None", markerfmt="red")

    axs[2].plot(n, semnal3_ex2(n))
    axs[2].stem(n_small, semnal3_ex2(n_small), basefmt="None", linefmt="None", markerfmt="red")
    plt.show()


def ex5():
    return 0


def ex6():
    return 0


if __name__ == "__main__":
    # ex1()
    # ex2()
    # ex3()
    ex5()
    ex6()

    # Exercitiul 4.
    # 40Hz-200Hz => max * 2 = 200 * 2 = 400Hz este minimul necesar (teorema esantionarii Nyquist-Shannon)

    #Exercitiul 7.
    # 80 = 10 * log10(90 / P_zgomot)
    # => P_zgomot = 90 / 10^8
