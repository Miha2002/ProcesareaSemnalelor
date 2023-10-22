import numpy as np
import matplotlib.pyplot as plt

## Exercitiul 1
def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)


def y(t):
    return np.cos(280 * np.pi * t + np.pi / 3)


def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)


def a_1():
    k = np.linspace(0, 0.03, 600)
    for i in k:
        plt.plot(i, 1, ".")
    plt.show()


def b_1():
    k = np.linspace(0, 0.03, 600)
    fig, axs = plt.subplots(3)
    axs[0].plot(k, x(k))
    axs[1].plot(k, y(k))
    axs[2].plot(k, z(k))
    plt.show()


def c_1():
    k = np.linspace(0, 0.03, 600)
    fig, axs = plt.subplots(3)
    axs[0].plot(k, x(k))
    axs[1].plot(k, y(k))
    axs[2].plot(k, z(k))

    k = np.linspace(0, 0.03, 6)
    axs[0].plot(k, x(k), ".")
    axs[1].plot(k, y(k), ".")
    axs[2].plot(k, z(k), ".")
    plt.show()


## Exercitiul 2

def f21(t):
    return 3 * np.sin(800 * np.pi * t + np.pi / 4)

def f22(t):
    return 5 * np.sin(1600 * np.pi * t + np.pi / 7)

def f23(t):
    return t - np.floor(t)

def f24(t):
    return np.sign(np.sin(2*np.pi*t))

def a_2():
    k = np.linspace(0, 4, 1600)
    plt.plot(k, f21(k))
    plt.show()

def b_2():
    k = np.linspace(0, 3, 2400)
    plt.plot(k, f22(k))
    plt.show()

def c_2():
    k = np.linspace(0, 5, 1200)
    plt.plot(k, f23(k))
    plt.show()

def d_2():
    k = np.linspace(0, 6, 1800)
    plt.plot(k, f24(k))
    plt.show()

def e_2():
    arr = np.random.randint(1000000, size=(128, 128))
    plt.imshow(arr)
    plt.show()

def f_2():
    arr = np.random.randint(2, size=(128, 128))
    plt.imshow(arr)
    plt.show()

    arr = np.zeros((128, 128))
    plt.imshow(arr)
    plt.show()

if __name__ == "__main__":
    # Exercitiul 1
    a_1()
    b_1()
    c_1()

    # Exercitiul 2
    a_2()
    b_2()
    c_2()
    d_2()
    e_2()
    f_2()

    # Exercitiul 3
    # Un semnal este digitizat cu o frecventa de esantionare de 2000 Hz.
    #
    # (a) Care este intervalul de timp intre doua esantioane?
    # 0.0005 = 1/2000
    #
    # (b) Daca un esantion este memorat pe 4 biti, cati bytes vor ocupa 1 ora de achizitie?
    # 2000 * 4 * 3600 (secunde/ora) / 8 = 3.600.000 bytes = 3.6 MBs
