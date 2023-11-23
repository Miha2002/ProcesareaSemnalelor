import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage

# X = misc.face(gray=True)
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()


def ex1_1(n1, n2):
    return np.sin(2*np.pi*n1 + 3*np.pi*n2)


def ex1_2(n1, n2):
    return np.sin(4*np.pi*n1) + np.cos(6*np.pi*n2)


def ex1_3(m1, m2):
    N = len(m1)
    arr = np.zeros(shape=(N, N))
    for elem1 in m1:
        for elem2 in m2:
            if elem1 == 0 and (elem2 == 5 or elem2 == N-5):
                arr[elem1][elem2] = 1
            else:
                arr[elem1][elem2] = 0
    return arr

def ex1_4(m1, m2):
    N = len(m1)
    arr = np.zeros(shape=(N,N))
    for elem1 in m1:
        for elem2 in m2:
            if elem2 == 0 and (elem1 == 5 or elem1 == N-5):
                arr[elem1][elem2] = 1
            else:
                arr[elem1][elem2] = 0
    return arr

def ex1_5(m1, m2):
    N = len(m1)
    arr = np.zeros(shape=(N, N))
    for elem1 in m1:
        for elem2 in m2:
            if elem1 == elem2 == 5:
                arr[elem1][elem2] = 1
            elif elem1 == elem2 == N-5:
                arr[elem1][elem2] = 1
            else:
                arr[elem1][elem2] = 0
    return arr


def ex1():
    n1 = np.arange(0,101,1,dtype=int)
    n2 = np.arange(0,101,1,dtype=int)

    image_ex1_3 = ex1_3(n1, n2)
    image_ex1_4 = ex1_5(n1, n2)
    image_ex1_5 = ex1_5(n1, n2)
    n1, n2 = np.meshgrid(n1, n2)
    image_ex1_1 = ex1_1(n1, n2)
    image_ex1_2 = ex1_2(n1, n2)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_ex1_1, cmap='viridis')
    plt.title('Imaginea pentru ex1_1')
    plt.subplot(1, 2, 2)
    plt.imshow(image_ex1_2, cmap='viridis')
    plt.title('Imaginea pentru ex1_2')
    plt.show()

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_ex1_3, cmap='viridis')
    plt.title('Imaginea pentru ex1_3')
    plt.subplot(1, 3, 2)
    plt.imshow(image_ex1_4, cmap='viridis')
    plt.title('Imaginea pentru ex1_4')
    plt.subplot(1, 3, 3)
    plt.imshow(image_ex1_5, cmap='viridis')
    plt.title('Imaginea pentru ex1_5')
    plt.show()

    # Spectrum
    spectrum_ex1_1 = np.abs(np.fft.fft2(image_ex1_1))
    spectrum_ex1_2 = np.abs(np.fft.fft2(image_ex1_2))
    spectrum_ex1_3 = np.abs(np.fft.fft2(image_ex1_3))
    spectrum_ex1_4 = np.abs(np.fft.fft2(image_ex1_4))
    spectrum_ex1_5 = np.abs(np.fft.fft2(image_ex1_5))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_1 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_1')
    plt.subplot(1, 2, 2)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_2 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_2')
    plt.show()

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_3 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_3')
    plt.subplot(1, 3, 2)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_4 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_4')
    plt.subplot(1, 3, 3)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_5 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_5')
    plt.show()



if __name__ == "__main__":
    ex1()