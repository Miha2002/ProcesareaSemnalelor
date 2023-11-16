import numpy as np
import time
import matplotlib.pyplot as plt

def semnal_ex1(n):
    return np.sin(2 * 25 * np.pi * n)

def ex1():
    valN = (128, 256, 512, 1024, 2048, 4096, 8192)
    t_f1=[]
    t_f2=[]

    for N in valN:
        n = np.linspace(0, N - 1, N)
        k = n.reshape((N,1))
        semnal = semnal_ex1(n)

        start = time.time() * 1000
        F = np.zeros((N, N), dtype=np.complex64)
        interm = np.exp(-2j * np.pi * k * n / N)
        F = np.sqrt(1/N) * np.dot(interm, semnal)
        end1 = time.time() * 1000

        F2 = np.fft.fft(semnal_ex1(n), N)
        # print(F2)
        end2 = time.time() * 1000


        t_f1.append(end1-start)
        t_f2.append(end2-end2)
        print("\nN=", N ,"\nOur version: ", end1-start, "ms\nNumpy version:", end2-end1,"ms")

    plt.figure(figsize=(10, 6))
    plt.plot(valN, t_f1, "o-", color="red")
    plt.plot(valN, t_f2, "o-")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    # N = 128
    # Our version: 4ms
    # Numpy version: 1ms
    #
    # N = 256
    # Our version: 13ms
    # Numpy version: 1ms
    #
    # N = 512
    # Our version: 51ms
    # Numpy version: 1ms
    #
    # N = 1024
    # Our version: 198ms
    # Numpy version: 2ms
    #
    # N = 2048
    # Our version: 809ms
    # Numpy version: 2ms
    #
    # N = 4096
    # Our version: 3271ms
    # Numpy version: 3ms
    #
    # N = 8192
    # Our version: 12980ms
    # Numpy version: 3ms


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


def ex6():
    rate, data = wavfile.read("vocale.wav")
    N = data.shape[0]
    k = N/100
    n = np.arange(0, N, k)
    n = [int(x) for x in n]

    for i in range(len(n)-1):
        x = data[n[i]: n[i+1]]
        col = np.fft.fft(x)
        matrix = np.hstack(col)

        if i+1 != len(n):
            x = data[n[i]+int(k/2): n[i + 1]+int(k//2)]
            col = np.fft.fft(x)
            matrix = np.hstack(col)

    # print(matrix)

    fs = rate
    nperseg = int(matrix.shape[0] / 100)
    noverlap = int(nperseg / 2)

    frequencies, times, spectrogram = signal.spectrogram(
        matrix.real, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    print(times)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, np.abs(np.log10(spectrogram)), shading='auto')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.show()


if __name__ == "__main__":
    # ex1()
    # ex2()
    # ex3()

    # Exercitiul 5.
    # Intre cateva dintre vocale se vad diferente, precum "a" si "e"
    # restul sunetelor sunt destul de similare

    ex6()

    # Exercitiul 4.
    # 40Hz-200Hz => max * 2 = 200 * 2 = 400Hz este minimul necesar (teorema esantionarii Nyquist-Shannon)

    #Exercitiul 7.
    # 80 = 10 * log10(90 / P_zgomot)
    # => P_zgomot = 90 / 10^8
