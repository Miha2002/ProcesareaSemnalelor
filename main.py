import numpy as np
import matplotlib.pyplot as plt


def semnal_sin(t):
    return np.sin(200 * np.pi * t + 3 * np.pi / 4)


def semnal_cos(t):
    return np.cos(200 * np.pi * t + np.pi / 4)


def ex1():
    k = np.linspace(0, 0.03, 600)

    fig, axs = plt.subplots(2)
    fig.suptitle("Ex.1")
    for i in k:
        axs[0].plot(i, semnal_sin(i), '.', color="gray")
        axs[1].plot(i, semnal_cos(i), '.', color="gray")
    plt.plot()


def f_ex2(t, faza):
    return 2 * np.sin(200 * np.pi * t + faza)


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def ex2():
    k = np.linspace(0, 0.03, 800)

    arr_ex2 = np.zeros(len(k))
    index = 0
    fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(6, 4))
    fig.suptitle("Ex.2")
    for i in k:
        axs1.plot(i, f_ex2(i, np.pi), '.', color="black")
        arr_ex2[index] = f_ex2(i, np.pi)
        index += 1
        axs1.plot(i, f_ex2(i, np.pi / 2), '.', color="blue")
        axs1.plot(i, f_ex2(i, np.pi / 4), '.', color="red")
        axs1.plot(i, f_ex2(i, np.pi / 6), '.', color="green")
    axs1.plot()

    z = np.random.normal(0, 1, len(k))
    SNR = 0.1
    gama = np.sqrt(np.power(normalize(arr_ex2, 0, 0.03), 2, dtype=complex) /
                   (np.power(normalize(z, 0, 0.3), 2, dtype=complex) * SNR))

    res = arr_ex2 + gama * z

    axs2.plot(k, res, '.', color="black")


if __name__ == '__main__':
    #     ex1()
    ex2()