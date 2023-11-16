import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
# import pandas as pd

def pctD():
    data = np.genfromtxt("Train.csv", delimiter=',', names=True, dtype=None, encoding=None)
    semnal = data['Count']
    N = len(semnal)
    f = 1/3600

    fft = np.fft.fft(semnal)
    res = np.abs(fft)
    n = np.fft.fftfreq(N, d=1/f)

    plt.plot(n, res)
    plt.xlabel("Frecventa(Hz)")
    plt.ylabel("Fourier")
    plt.show()


def pctE():
    data = np.genfromtxt("Train.csv", delimiter=',', names=True, dtype=None, encoding=None)
    time = pd.to_datetime(data["Datetime"], format='%d-%m-%Y %H:%M', errors='coerce')
    semnal = data['Count']
    N = len(semnal)
    f = 1 / 3600

    res = semnal.mean()
    print(int(res))
    if res != 0:
        print("Semnalul are o componenta continua cu valoarea:", res)

        semnal_nou = int((semnal - res).mean()) # foarte aproape de 0, erori ici colo idk
        print("Semnalul nou are componenta continua:", semnal_nou, " ,adica nu are :)))")
    else:
        print("Semnalul nu are componenta continuă.")


def pctF():
    data = np.genfromtxt("Train.csv", delimiter=',', names=True, dtype=None, encoding=None)
    semnal = data['Count']
    N = len(semnal)
    f = 1 / 3600

    res_fft = np.fft.fft(semnal)
    n = np.fft.fftfreq(N, d=1 / f)

    max_val = np.argsort(np.abs(res_fft))[-4:]
    max_frecv = n[max_val]

    for i, val in enumerate(max_val):
        frequency = max_frecv[i]
        amplitude = np.abs(res_fft[val])
        print("Frecventa: ", frequency, " Hz,  Amplitudine:", amplitude)


if __name__ == "__main__":

    # Punctul A.
    # 18288 esantioane, 1 esantion / ora
    # intr-o secunda => 1/(60*60) = 0.000277 Hz
    # data = np.genfromtxt("Train.csv", delimiter=',', names=True, dtype=None, encoding=None)
    # time = pd.to_datetime(data["Datetime"], format='%d-%m-%Y %H:%M', errors='coerce')

    # Punctul B.
    # Considerand ca se precizeaza ca fiecare esantion reprezinta datele obtinute o data pe ora
    # inseamna ca cele 18288 esantioane au fost obtiute in 762 de zile
    # 18288 / 24 = 762

    # Punctul C.
    # Conform Nyquist frecventa maxima este 1/2 din frecventa de esantionare
    # => 0.000277 Hz / 2 = 0.000138 Hz

    pctD()
    pctE()
    pctF()
