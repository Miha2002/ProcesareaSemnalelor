import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

N = 1000
t = np.arange(N)
trend = 10 * t ** 2 + 4 * t
sezon = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 20)
variatii = np.random.normal(0, 2, N)

serie_timp = trend + sezon + variatii


def mediere_exponentiala(x, alpha):
    N = len(x)
    s = np.zeros(N)

    for t in range(N):
        s[t] = alpha * np.sum((1 - alpha)**(t-k) * x[k] for k in range(t+1)) + (1 - alpha)**t * x[0]

    return s

def ex2():
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
    min_error = float('inf')
    optimal_alpha = None

    for alpha in alpha_values:
        serie_mediere_exp = mediere_exponentiala(serie_timp, alpha)
        error = np.sum((serie_timp - serie_mediere_exp) ** 2)

        if error < min_error:
            min_error = error
            optimal_alpha = alpha

    plt.figure(figsize=(12, 6))

    plt.plot(t, serie_timp, '.', label='Seria de Timp')
    plt.plot(t, mediere_exponentiala(serie_timp, optimal_alpha), label='Mediere Exponențială')
    plt.legend()
    plt.show()

    print("Valoare optimă pentru alpha: ",optimal_alpha)

# Nu stiu ce fac aici fr
def ex3():
    q = 2
    theta_values = np.random.normal(0, 0.5, q)
    epsilon = np.random.normal(0, 100, N)
    serie_ma = np.zeros(N)

    for k in range(q, N):
        serie_ma[k] = epsilon[k] + np.dot(theta_values, epsilon[k - q:k][::-1])

    plt.figure(figsize=(12, 6))

    plt.plot(t, serie_timp, label='Seria de Timp Originală')
    plt.plot(t, serie_ma, label=f'Model MA (q={q})')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ex2()
    ex3()
