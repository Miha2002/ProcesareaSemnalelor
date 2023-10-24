import time

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd


def semnal_sin(t):
    return np.sin(200 * np.pi * t + 3 * np.pi / 4)


def semnal_cos(t):
    return np.cos(200 * np.pi * t + np.pi / 4)


def ex1():
    k = np.linspace(0, 0.03, 600)
    fig, axs = plt.subplots(2)

    axs[0].plot(k, semnal_sin(k))
    axs[1].plot(k, semnal_cos(k))
    plt.show()


def ex2_f1(t, faza):
    return 2 * np.sin(200 * np.pi * t + faza)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def ex2():
    k = np.linspace(0, 0.03, 800)

    plt.plot(k, ex2_f1(k, np.pi))
    plt.plot(k, ex2_f1(k, np.pi / 2), color="blue")
    plt.plot(k, ex2_f1(k, np.pi / 4), color="red")
    plt.plot(k, ex2_f1(k, np.pi / 6), color="green")
    plt.show()

    arr_ex2 = np.zeros(len(k))
    index = 0
    for i in k:
        arr_ex2[index] = ex2_f1(i, np.pi)
        index += 1

    z = np.random.normal(0, 1, len(k))
    snr = 100
    gama = np.sqrt(np.power(normalize(arr_ex2), 2) /
                   (np.power(normalize(z), 2) * snr))
    res = arr_ex2 + gama * z

    plt.plot(k, res)
    plt.show()


def ex3():
    fs = 44100
    rate, ex2_a = scipy.io.wavfile.read("ex2_a.wav")
    rate, ex2_b = scipy.io.wavfile.read("ex2_b.wav")
    rate, ex2_c = scipy.io.wavfile.read("ex2_c.wav")  # frecventa prea inalta
    rate, ex2_d = scipy.io.wavfile.read("ex2_d.wav")  # frecventa prea inalta

    sd.play(ex2_a, fs)
    time.sleep(2)
    sd.play(ex2_b, fs)
    time.sleep(2)
    # sd.play(ex2_c, fs)
    # time.sleep(2)
    # sd.play(ex2_d, fs)
    # time.sleep(2)

def semnal_s(t):
    return 3 * np.sin(800 * np.pi * t + np.pi / 4)

def semnal_st(t):
    return t - np.floor(t)

def ex4():
    k = np.linspace(0, 5, 2000)
    arr_sin = semnal_s(k)
    arr_st = semnal_st(k)
    arr_res = np.add(arr_sin,arr_st)

    # plt.plot(k, arr_sin, color="green")
    # plt.plot(k, arr_st, color="blue")
    plt.plot(k, arr_res, color="red")
    plt.show()

def semnal_ex5(t,frecventa):
    return np.sin(2 * frecventa * np.pi * t)

def ex5():
    k = np.linspace(0, 4, 44100*4)
    arr_fv1 = semnal_ex5(k,440)
    arr_fv2 = semnal_ex5(k,920)

    arr_res = np.concatenate((arr_fv1,arr_fv2), axis=None)
    k = np.concatenate((k,k), axis=None)

    sd.play(arr_res, 44100)
    sd.wait()

    # cu cat frecventa este un numar mai mare => frecventa este mai inalta
    # => sunetul este mai ascutit
    # se aude clar unde se face schimbarea dintre cele 2 frecvente


def semnal_ex6(t,frecventa):
    return np.sin(2 * frecventa * np.pi * t)

def ex6():
    fv = 400/2
    k = np.linspace(0, 4, 400)
    plt.plot(k, semnal_ex6(k,fv))
    fv = 400/4
    plt.plot(k, semnal_ex6(k, fv))
    fv = 0
    plt.plot(k, semnal_ex6(k, fv))

    plt.show()

    # cu cat frecventa este mai mica, cu atat este mai plat graficul functiei sinusoidale
    # pentru frecventa = 0Hz, este constant 0


def semnal_ex7(t):
    return np.sin(2 * 1000 * np.pi * t)

def ex7():
    k = np.linspace(0,1,200)
    plt.plot(k, semnal_ex7(k))
    plt.show()

    res = semnal_ex7(k)[::4]
    plt.plot(np.linspace(0,1,len(res)),res)
    plt.show()

    res2 = semnal_ex7(k)[1::4]
    plt.plot(np.linspace(0, 1, len(res2)), res2)
    plt.show()

    # graficul iese mai smooth cand primeste vectorul initial
    # diferenta dintre cele 2 optiuni este o shiftare la dreapta a punctelor
    # este vizibil, dar nicio varianta nu este mai buna ca cealalta

if __name__ == "__main__":
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
    ex6()
    ex7()