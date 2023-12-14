import numpy as np
import matplotlib.pyplot as plt

def ex1():
# subpunctul A.
    N = 1000
    t = np.arange(N)
    trend = 10 * t**2 + 4 * t
    sezon = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 20)
    variatii = np.random.normal(0, 2, N)

    serie_timp = trend + sezon + variatii

    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    plt.plot(t, serie_timp, label='Seria de Timp')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t, trend, label='Trend')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(t, sezon, label='Sezon')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(t, variatii, label='Variatii Mici')
    plt.legend()
    plt.show()

# subpunctul B.
    autocorrelation = np.correlate(serie_timp, serie_timp, mode='full')
    autocorrelation /= np.max(autocorrelation)
    lags = np.arange(-N + 1, N)

    plt.figure(figsize=(10, 5))
    plt.stem(lags, autocorrelation, use_line_collection=True)
    plt.title('Autocorelatie a Seriei de Timp')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrrelation')
    plt.show()

# subpunctul C.

if __name__ == '__main__':
    ex1()