import numpy as np
import matplotlib.pyplot as plt


## exercitiul 1
mu = 0
sigma = 1
data = np.random.normal(mu, sigma, 1000)
plt.hist(data, bins=30, density=True, alpha=0.7, color='blue')

# Afișează densitatea de probabilitate pentru distribuția normală
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
plt.plot(x, p, 'k', linewidth=2)

plt.title('Distribuția normală unidimensională')
plt.xlabel('Valoare')
plt.ylabel('Densitatea de probabilitate')
plt.show()

## bidimensional

cov_matrix = np.array([[1, 3/5], [3/5, 1]])
data = np.random.multivariate_normal([0, 0], cov_matrix, 1000)
plt.scatter(data[:, 0], data[:, 1], alpha=0.7, color='blue')

plt.title('Distribuție Gaussiană Bidimensională')
plt.xlabel('Variabila 1')
plt.ylabel('Variabila 2')
plt.show()

