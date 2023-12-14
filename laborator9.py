import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

def ex1():
    N = 1000
    t = np.arange(N)
    trend = 10 * t ** 2 + 4 * t
    sezon = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 20)
    variatii = np.random.normal(0, 2, N)

    serie_timp = trend + sezon + variatii
    # plt.plot(serie_timp)
    # plt.show()

    # order = (1, 0.5)
    # results = ArmaProcess(serie_timp, order)
    # print(results)
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)
    # ax.plot(results.generate_sample(nsample=N*10))
    # plt.show()

    alpha = 0.5
    result = np.zeros(N)

    for j in t:
        for i in range(j):
            result[j] = result[j] + (1-alpha)**(j-i) * serie_timp[i] + (1-alpha)**j * serie_timp[0]
        result[j] = result[j] * alpha

    plt.plot(result)
    plt.show()


if __name__ == '__main__':
    ex1()